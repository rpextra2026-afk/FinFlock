"""Tessera local loader with enforced train/held-out separation.

Source-specific rights and attribution fields are retained as provenance.
This loader does not assign a new license to upstream source material.
"""
from pathlib import Path

import pandas as pd

TRAIN_SPLITS = {"dev"}
HELDOUT_SPLITS = {"embargo", "test"}
SPLIT_COLS = ("split_policy", "split", "partition")
CATEGORY_DIRS = {"financial": "financial", "geopolitical": "geopolitical", "social": "social_reddit"}
UNSUPPORTED_FIELDS = {"alias_learned_finance_headline", "structured_ticker_current_only"}


class SplitGuardError(ValueError):
    """held-out (embargo/test) or unverifiable rows were passed to a training input"""


def assert_dev_only(df: pd.DataFrame, name: str = "input") -> pd.DataFrame:
    cols = [c for c in SPLIT_COLS if c in df.columns]
    if not cols:
        raise SplitGuardError(f"{name}: no split column ({'/'.join(SPLIT_COLS)}); cannot verify it is dev-only, refusing to train on it")
    for c in cols:
        v = df[c].astype(str)
        bad = ~(v.isin(TRAIN_SPLITS) | v.eq("train"))
        if bad.any():
            vals = sorted(set(v[bad]))
            raise SplitGuardError(f"{name}: {int(bad.sum()):,} row(s) with {c} in {vals} are not dev/train; embargo and test rows must never reach training")
    return df


def _root(pkg) -> Path:
    p = Path(pkg)
    if not (p / "MANIFEST.json").exists():
        raise FileNotFoundError(f"{p} is not a Tessera package folder (no MANIFEST.json)")
    return p


def supported_pins(pins: pd.DataFrame) -> pd.DataFrame:
    return pins[pins["confidence"].isin(["high", "medium"]) & ~pins["source_field"].isin(UNSUPPORTED_FIELDS)]


def add_asset_labels(items: pd.DataFrame, pins: pd.DataFrame, id_col: str = "item_id") -> pd.DataFrame:
    """company_labelled / country_linked / commodity_linked come only from supported pins; asset_label_basis says which kind."""
    sp = supported_pins(pins)
    by = {t: set(sp.loc[sp["pin_type"] == t, id_col if id_col in sp.columns else "item_id"]) for t in ("company", "country", "commodity")}
    out = items.copy()
    out["company_labelled"] = out[id_col].isin(by["company"])
    out["country_linked"] = out[id_col].isin(by["country"])
    out["commodity_linked"] = out[id_col].isin(by["commodity"])
    out["asset_label_basis"] = "none"
    out.loc[out["country_linked"] | out["commodity_linked"], "asset_label_basis"] = "country_or_commodity_only"
    out.loc[out["company_labelled"], "asset_label_basis"] = "supported_company_pin"
    return out


def load_items(pkg, categories=("financial", "geopolitical", "social"), include_supplement=True, exclude_template_crossing=False, with_text=True, _allow_heldout=None):
    """Train-partition items (+text) with asset labels. exclude_template_crossing drops train Federal Register/OFAC notices whose title+abstract template also occurs in held-out data
    (they are distinct notices and are kept by default; set True for a template-strict run)."""
    root = _root(pkg)
    frames = []
    for cat in categories:
        d = root / CATEGORY_DIRS[cat]
        it = pd.read_parquet(d / "items.parquet")
        it = it[it["partition"].eq("train")] if _allow_heldout is None else it[it["partition"].eq(_allow_heldout)]
        if exclude_template_crossing and "template_crosses_split" in it.columns:
            it = it[~it["template_crosses_split"].fillna(False)]
        if with_text:
            tx = pd.read_parquet(d / "text.parquet")
            it = it.merge(tx, on="item_id", how="left")
        pn = pd.read_parquet(d / "pins.parquet")
        it = add_asset_labels(it, pn)
        frames.append(it)
    if include_supplement:
        rec = pd.read_parquet(root / "geo_supplement_records" / "records.parquet")
        rec = rec[rec["partition"].eq("train")] if _allow_heldout is None else rec[rec["partition"].eq(_allow_heldout)]
        s = pd.DataFrame({"item_id": rec["record_id"], "source_id": rec["source_id"], "category": "geo_supplement", "event_id": rec["event_id"], "family": "geo_supplement", "published_utc": pd.to_datetime(rec["published_date"], utc=True),
                          "available_utc": rec["available_utc"], "policy_session": rec["policy_session"], "split": rec["split"], "partition": rec["partition"], "text_kind": "news_text", "title": rec["headline"], "source_url": rec["canonical_url"]})
        if with_text:
            tx = pd.concat([pd.read_parquet(root / "supplement_text" / "text.parquet"), pd.read_parquet(root / "geo_supplement_records" / "public_text.parquet")[["record_id", "text"]]])
            s = s.merge(tx.rename(columns={"record_id": "item_id"}), on="item_id", how="left")
        pn = pd.read_parquet(root / "geo_supplement_records" / "pins.parquet").rename(columns={"record_id": "item_id"})
        frames.append(add_asset_labels(s, pn))
    df = pd.concat(frames, ignore_index=True)
    if _allow_heldout is None:
        assert_dev_only(df, "training items")
    return df


def load_gdelt_panel(pkg, purpose: str = "train", split: str | None = None) -> pd.DataFrame:
    """purpose='train': only split_policy == 'dev' rows. purpose='eval': requires split='embargo' or 'test' (never mixed with training)."""
    root = _root(pkg)
    p = pd.read_parquet(root / "common" / "gdelt_country_day.parquet")
    if purpose == "train":
        p = p[p["split_policy"].eq("dev")]
        assert_dev_only(p, "GDELT panel (training)")
        return p
    if purpose == "eval":
        if split not in HELDOUT_SPLITS:
            raise SplitGuardError("GDELT panel eval load needs split='embargo' or 'test'")
        return p[p["split_policy"].eq(split)]
    raise ValueError("purpose must be 'train' or 'eval'")


def load_training_inputs(pkg, exclude_template_crossing=False, **kw) -> dict:
    root = _root(pkg)
    items = load_items(root, exclude_template_crossing=exclude_template_crossing, **kw)
    ent = pd.read_parquet(root / "common" / "entities.parquet")
    return {"items": items, "gdelt_panel": load_gdelt_panel(root, "train"), "entities": ent}


def load_heldout(pkg, split: str = "test", categories=("financial", "geopolitical", "social")) -> pd.DataFrame:
    """evaluation data only; split in {'embargo','test'} (partition heldout_embargo / heldout_test). Refuses training splits."""
    if split not in HELDOUT_SPLITS:
        raise SplitGuardError(f"load_heldout refuses split={split!r}; use 'embargo' or 'test'")
    return load_items(pkg, categories=categories, _allow_heldout=f"heldout_{split}")


def guard_training_batch(*frames, names=None):
    """call on every table before it enters a training run; raises SplitGuardError on the first held-out or unverifiable table"""
    for i, f in enumerate(frames):
        assert_dev_only(f, (names[i] if names else f"input[{i}]"))
    return frames


def render_reader_input(row, max_chars: int | None = None) -> str:
    """the text a reader model would receive: source, date, title, body. SEC text is capped at 6,000 characters upstream and starts at the first 'Item N.NN'; it is not news-style prose."""
    body = str(row.get("text") or "")
    if max_chars:
        body = body[:max_chars]
    return f"[{row.get('source_id')}] {str(row.get('published_utc'))[:10]}\n{row.get('title') or ''}\n\n{body}".strip()
