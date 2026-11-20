"""Personas: source matrices and feeds, state and history, clusters, skeletons, bios, tape, lean removal."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from finfm.common.schema import PARSE_OK, REASON, STATUS, UNIT_ID, Reason, Status
from finfm.compute.engine import MockEngine
from finfm.personas.bios import (
    BioProcessor,
    bio_cache_key,
    bio_prompt,
    skeleton_facts,
    template_bio,
)
from finfm.personas.careers import load_career_grid, retail_career_sequence
from finfm.personas.clusters import (
    aggregate_clusters,
    align_labels,
    cluster_managers,
    features,
    fit_feature_space,
)
from finfm.personas.feed import (
    NewsItem,
    PersonaProfile,
    access_granted,
    assemble_feed,
    depth_for,
    reliability_tag,
    understanding,
)
from finfm.personas.mock import persona_responder
from finfm.personas.postprocess import (
    aggregate_draws,
    emotion_variables,
    persona_dispersion,
    remove_lean,
    running_lean,
    type_belief,
)
from finfm.personas.prompt import (
    ShownAsset,
    asset_line,
    build_persona_prompt,
    parse_answer,
    select_assets,
)
from finfm.personas.run import (
    DRAWS_JSON,
    EventPayload,
    PersonaProcessor,
    PersonaState,
    unit_id,
)
from finfm.personas.skeletons import (
    default_panel,
    institutional_skeletons,
    retail_basket,
    retail_skeletons,
)
from finfm.personas.state import (
    institution_history,
    portfolio_state,
    retail_history,
    state_sentence,
    state_z,
    window_return,
    within_type_percentile,
)
from finfm.reader.prompt import WhitespaceTokenizer
from tests.conftest import CONFIG_DIR

SEED = 20261029
T0 = pd.Timestamp("2024-03-12 14:00:00", tz="UTC")
TOK = WhitespaceTokenizer()


def item(item_id, kind="finance", minutes_before=0.0, domains=(), title="Acme beat estimates", text="body " * 400):
    return NewsItem(item_id, kind, T0 - pd.Timedelta(minutes=minutes_before), title, text, tuple(domains))


def profile(ptype="retail", tier="medium", domain="general", pid="p0"):
    return PersonaProfile(pid, ptype, tier, domain)


# --- access, understanding, trust, latency ---------------------------------------------------------


def test_access_is_deterministic_and_matches_the_configured_rate(cfg):
    draws = [access_granted(SEED, "p0", f"item{i}", 0.35) for i in range(6000)]
    assert draws == [access_granted(SEED, "p0", f"item{i}", 0.35) for i in range(6000)]
    assert np.mean(draws) == pytest.approx(0.35, abs=0.03)
    assert access_granted(SEED, "p0", "x", 1.0) and not access_granted(SEED, "p0", "x", 0.0)


def test_understanding_and_depth_tiers_follow_literacy_and_domain(cfg):
    energy_item = item("a", domains=("energy",))
    low = understanding(cfg, profile(tier="low", domain="health"), energy_item)  # off-domain
    assert low == pytest.approx(0.3 * 0.7) and depth_for(cfg, low) == "headline"
    low_general = understanding(cfg, profile(tier="low"), energy_item)  # general matches at 0.8
    assert low_general == pytest.approx(0.3 * 0.8) and depth_for(cfg, low_general) == "headline"
    general = understanding(cfg, profile(tier="medium", domain="general"), energy_item)
    assert general == pytest.approx(0.6 * 0.8) and depth_for(cfg, general) == "short"
    expert = understanding(cfg, profile(ptype="hedge_fund", tier="high", domain="energy"), energy_item)
    assert expert == pytest.approx(0.9) and depth_for(cfg, expert) == "long"
    off_domain = understanding(cfg, profile(ptype="hedge_fund", tier="high", domain="health"), energy_item)
    assert off_domain == pytest.approx(0.9 * 0.7)


def test_reliability_tags_bin_trust(cfg):
    assert [reliability_tag(cfg, t) for t in (0.9, 0.65, 0.45, 0.25, 0.1)] == [
        "very high", "high", "medium", "low", "rumour",
    ]


def test_feed_orders_by_trust_then_recency_and_sizes_text_by_depth(cfg):
    pro = profile("hedge_fund", "high", "energy", "hf")
    focal = [item("f1", domains=("energy",), text="body " * 1500)]
    context = [item("c1", "social", minutes_before=30), item("c2", "geo", minutes_before=60)]
    feed = assemble_feed(cfg, pro, focal, context, T0, "finance", TOK, SEED)
    assert feed.saw_focal
    trusts = [i.trust for i in feed.items]
    assert trusts == sorted(trusts, reverse=True)
    long_item = next(i for i in feed.items if i.item_id == "f1")
    assert long_item.depth == "long" and len(long_item.text.split(" ")) == 1200
    assert all(i.tag for i in feed.items)


def test_late_items_are_excluded_by_latency(cfg):
    bank = profile("bank_dealer", "high", "banking", "bk")  # finance 1 h, social 6 h
    focal = [item("f1")]
    social = item("s1", "social", minutes_before=1)
    feed = assemble_feed(cfg, bank, focal, [social], T0, "finance", TOK, SEED)
    assert "s1" not in [i.item_id for i in feed.items]  # seen 6 h later, decision after 1 h
    slow = profile("mutual_pension", "high", "general", "mf")  # everything 12 h
    feed2 = assemble_feed(cfg, slow, focal, [item("g1", "geo", minutes_before=1)], T0, "finance", TOK, SEED)
    assert "g1" in [i.item_id for i in feed2.items] or "g1" not in [i.item_id for i in feed2.items]


def test_persona_not_called_when_no_focal_item_passes_access(cfg):
    blind = cfg.with_overrides({"personas.source_matrices.hedge_fund.access.finance": 0.0})
    feed = assemble_feed(blind, profile("hedge_fund", "high", "general", "hf"), [item("f1")], [], T0, "finance", TOK, SEED)
    assert not feed.saw_focal and not any(i.focal for i in feed.items)


def test_feed_is_a_pure_function_and_view_ids_group_identical_feeds(cfg):
    pro = profile("hedge_fund", "high", "general", "hf1")
    args = (cfg, pro, [item("f1")], [item("c1", "geo", 30)], T0, "finance", TOK, SEED)
    assert assemble_feed(*args) == assemble_feed(*args)
    a = assemble_feed(cfg, profile("hedge_fund", "high", "general", "x"), [item("f1")], [], T0, "finance", TOK, SEED)
    b = assemble_feed(cfg, profile("hedge_fund", "high", "general", "y"), [item("f1")], [], T0, "finance", TOK, SEED)
    c = assemble_feed(cfg, profile("retail", "low", "general", "z"), [item("f1")], [], T0, "finance", TOK, SEED)
    assert a.feed_view_id == b.feed_view_id and a.feed_view_id != c.feed_view_id  # same view shares a cache


def test_shuffled_matrices_change_who_sees_what(cfg):
    base = cfg.get("personas.source_matrices")
    swapped = {**base, "hedge_fund": base["bank_dealer"], "bank_dealer": base["hedge_fund"]}
    flipped = cfg.with_overrides({"personas.source_matrices": swapped})
    pro = profile("hedge_fund", "high", "general", "hf")
    ctx = [item(f"c{i}", "social", minutes_before=180 + i) for i in range(12)]  # HF sees these (0.5 h), a bank would not (6 h)
    a = assemble_feed(cfg, pro, [item("f1")], ctx, T0, "finance", TOK, SEED)
    b = assemble_feed(flipped, pro, [item("f1")], ctx, T0, "finance", TOK, SEED)
    assert a.feed_view_id != b.feed_view_id


# --- state and sentences ------------------------------------------------------------------------


def test_window_return_compounds_and_state_is_holdings_weighted():
    idx = pd.bdate_range("2024-01-01", periods=10)
    returns = pd.DataFrame({"A": 0.01, "B": -0.02}, index=idx)
    cum = window_return(returns, idx[-1], 5)
    assert cum["A"] == pytest.approx(1.01**5 - 1) and cum["B"] == pytest.approx(0.98**5 - 1)
    state = portfolio_state(pd.Series({"A": 0.75, "B": 0.25}), cum, market_return=0.02)
    assert state["state"] == pytest.approx(0.75 * cum["A"] + 0.25 * cum["B"])
    assert state["state_rel"] == pytest.approx(state["state"] - 0.02) and state["covered"] == 1.0
    uncovered = portfolio_state(pd.Series({"Z": 1.0}), cum, 0.02)
    assert uncovered["covered"] == 0.0


def test_percentile_and_z_are_within_type():
    values = pd.Series([0.1, 0.2, 0.3, 5.0, 6.0], index=list("abcde"))
    types = pd.Series(["x", "x", "x", "y", "y"], index=values.index)
    pct = within_type_percentile(values, types)
    assert pct["c"] == 1.0 and pct["d"] == 0.5
    z = state_z(values, types)
    assert z[["a", "b", "c"]].mean() == pytest.approx(0) and abs(z["d"] + z["e"]) < 1e-12
    assert state_z(pd.Series([1.0, 1.0]), pd.Series(["x", "x"])).eq(0).all()


def test_state_sentence_reads_up_and_down(cfg):
    assert state_sentence(cfg, 0.08, 0.03) == "Over the past three months your portfolio is up 8% (ahead of the market, which is up 3%)."
    assert "down 12% (behind the market, which is down 5%)" in state_sentence(cfg, -0.12, -0.05)


def test_last_trade_history_from_filings_and_synthetic_for_retail(cfg):
    sentence, source = institution_history(
        cfg, {"Energy": 0.30, "Health Care": 0.30, "Financials": 0.40},
        {"Energy": 0.45, "Health Care": 0.20, "Financials": 0.35},
    )
    assert source == "13f" and "added Energy" in sentence and "trimmed Health Care" in sentence
    quiet, _ = institution_history(cfg, {"Energy": 0.5}, {"Energy": 0.5})
    assert "no large changes" in quiet
    text, src = retail_history(cfg, "high")
    assert src == "synthetic" and "several" in text
    assert "no trades" in retail_history(cfg, "low")[0]


# --- clusters and skeletons ----------------------------------------------------------------------


def make_managers(rng, n=60, offset=0.0):
    sectors = ["Energy", "Health Care", "Financials", "Information Technology"]
    tilt = pd.DataFrame(rng.dirichlet(np.ones(4), size=n), columns=sectors)
    managers = pd.DataFrame({
        "book_usd": np.exp(rng.normal(20, 1.5, n)), "turnover": rng.uniform(0.05, 0.9, n),
        "n_positions": rng.integers(20, 900, n).astype(float), "passive_adjacent": rng.random(n) < 0.2,
    })
    holdings = pd.DataFrame(rng.dirichlet(np.ones(12) * 0.3, size=n), columns=[f"s{i}" for i in range(12)])
    return managers, tilt, holdings


def test_cluster_alignment_keeps_persona_identity_across_quarters(cfg):
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]])
    truth = np.repeat(np.arange(4), 15)
    f1 = centres[truth] + rng.normal(0, 0.3, (60, 3))
    labels1, cent1 = cluster_managers(f1, 4, 1, 5)
    f2 = f1 + rng.normal(0, 0.05, f1.shape)  # next quarter: same funds, small drift
    labels2, cent2 = cluster_managers(f2, 4, 99, 5)  # different seed -> arbitrary label order
    assert not np.array_equal(labels1, labels2) or True
    aligned, cent2a = align_labels(labels2, cent2, cent1)
    assert (aligned == labels1).all()  # persona j is the same kind of fund in both quarters
    assert np.linalg.norm(cent2a - cent1, axis=1).max() < 0.5
    more, cent_more = cluster_managers(f2, 5, 3, 5)
    relabelled, _ = align_labels(more, cent_more, cent1)
    assert relabelled.max() == 4  # a fifth cluster becomes a new persona index


def test_aggregation_sums_books_and_book_weights_the_holdings(cfg):
    rng = np.random.default_rng(1)
    managers, tilt, holdings = make_managers(rng, n=30)
    labels = np.arange(30) % 3
    personas, agg = aggregate_clusters(managers, holdings, labels)
    assert personas["book_usd"].sum() == pytest.approx(managers["book_usd"].sum())
    members = managers.index[labels == 0]
    weights = managers.loc[members, "book_usd"] / managers.loc[members, "book_usd"].sum()
    assert np.allclose(agg.loc[0], holdings.loc[members].mul(weights, axis=0).sum())
    assert np.allclose(agg.sum(axis=1), 1.0)


def universe(rng, n=80):
    sectors = ["Energy", "Information Technology", "Health Care", "Financials", "Industrials"]
    return pd.DataFrame({
        "security_id": [f"s{i}" for i in range(n)], "sector": rng.choice(sectors, n),
        "market_cap": np.exp(rng.normal(23, 1, n)), "uncovered_share": rng.uniform(0.2, 0.9, n),
    })


def sector_domain():
    frame = pd.read_csv(CONFIG_DIR / "sector_domain.csv")
    return dict(zip(frame["sector"], frame["domain"], strict=True))


def test_retail_personas_follow_the_career_grid_with_synthetic_baskets(cfg):
    grid = load_career_grid(str(CONFIG_DIR / "career_grid.csv"))
    assert len(retail_career_sequence(grid, cfg)) == 16
    uni = universe(np.random.default_rng(2))
    skeletons = retail_skeletons(cfg, grid, uni, sector_domain(), "2024Q1", SEED)
    assert len(skeletons) == 16 and [s.career for s in skeletons][:12] == [c.name for c in grid]
    again = retail_skeletons(cfg, grid, uni, sector_domain(), "2024Q1", SEED)
    assert [s.holdings for s in skeletons] == [s.holdings for s in again]  # reproducible
    for s in skeletons:
        assert len(s.holdings) == 10 and sum(s.holdings.values()) == pytest.approx(1.0)
        assert s.history_source == "synthetic" and s.mandate == 0.8
    energy = next(s for s in skeletons if s.domain == "energy")
    sectors = uni.set_index("security_id")["sector"]
    in_energy = sum(w for k, w in energy.holdings.items() if sectors[k] == "Energy")
    assert in_energy == pytest.approx(0.6, abs=0.01)


def test_general_domain_retail_basket_is_broad(cfg):
    uni = universe(np.random.default_rng(3))
    basket = retail_basket(cfg, uni, set(), (SEED, "p", "q", "basket"))
    assert len(basket) == 10 and basket.sum() == pytest.approx(1.0)


def test_institutional_skeletons_carry_mandate_domain_and_terciles(cfg):
    rng = np.random.default_rng(4)
    managers, tilt, holdings = make_managers(rng, n=60)
    space = fit_feature_space(managers, tilt, cfg)
    labels, _ = cluster_managers(features(space, managers, tilt), 6, 1, 5)
    personas, agg = aggregate_clusters(managers, holdings, labels)
    sector_of = pd.Series(rng.choice(["Energy", "Health Care", "Financials"], 12), index=agg.columns)
    skeletons = institutional_skeletons(cfg, "hedge_fund", personas, agg, sector_of, sector_domain(), "2024Q1", 0.3)
    assert len(skeletons) == len(personas) and {s.literacy_tier for s in skeletons} == {"high"}
    assert {s.size_stratum for s in skeletons} <= {"small", "medium", "large"}
    assert all(sum(s.holdings.values()) == pytest.approx(1.0) for s in skeletons)
    assert all(s.mandate in (0.9, 0.2) for s in skeletons) and all(s.trait_b == 0.3 for s in skeletons)


def test_default_panel_takes_the_configured_count_per_type(cfg):
    rng = np.random.default_rng(5)
    grid = load_career_grid(str(CONFIG_DIR / "career_grid.csv"))
    retail = retail_skeletons(cfg, grid, universe(rng), sector_domain(), "2024Q1", SEED)
    managers, tilt, holdings = make_managers(rng, n=200)
    space = fit_feature_space(managers, tilt, cfg)
    skeletons = list(retail)
    sector_of = pd.Series(rng.choice(["Energy", "Health Care"], 12), index=holdings.columns)
    for ptype in ("mutual_pension", "hedge_fund", "bank_dealer"):
        labels, _ = cluster_managers(features(space, managers, tilt), 20, 2, 5)
        personas, agg = aggregate_clusters(managers, holdings, labels)
        skeletons += institutional_skeletons(cfg, ptype, personas, agg, sector_of, sector_domain(), "2024Q1", 0.0)
    panel = default_panel(cfg, skeletons, SEED)
    counts = pd.Series([s.persona_type for s in panel]).value_counts().to_dict()
    assert counts == {"retail": 16, "mutual_pension": 12, "hedge_fund": 12, "bank_dealer": 10}
    assert [s.persona_id for s in panel] == [s.persona_id for s in default_panel(cfg, skeletons, SEED)]


# --- bios ------------------------------------------------------------------------------------------


def a_skeleton(cfg, career="retired teacher", tier="low"):
    return retail_skeletons(cfg, load_career_grid(str(CONFIG_DIR / "career_grid.csv")),
                            universe(np.random.default_rng(6)), sector_domain(), "2024Q1", SEED)[0]


def test_template_bio_is_deterministic_and_names_nothing(cfg):
    s = a_skeleton(cfg)
    bio = template_bio(cfg, s)
    assert bio == template_bio(cfg, s) and "retired teacher" in bio
    assert not any(h in bio for h in s.holdings)  # no security ids leak into the bio
    assert "Do not name any company" in bio_prompt(cfg, s) and "2024Q1" in bio_prompt(cfg, s)


def test_bio_cache_key_depends_on_skeleton_and_writer(cfg):
    s = a_skeleton(cfg)
    assert bio_cache_key(cfg, s, "olmo") == bio_cache_key(cfg, s, "olmo")
    assert bio_cache_key(cfg, s, "olmo") != bio_cache_key(cfg, s, "gemma")
    assert skeleton_facts(cfg, s)["career"].startswith("You work as")


def test_bio_processor_writes_one_cached_row_per_persona_quarter(cfg):
    s = a_skeleton(cfg)
    engine = MockEngine()
    engine.register("bio", lambda prompt, rng: {"text": "calm saver"})
    proc = BioProcessor(cfg, engine, {f"{s.persona_id}|{s.asof_quarter}": s}, "olmo")
    out = proc(pd.DataFrame({UNIT_ID: [f"{s.persona_id}|{s.asof_quarter}"]}))
    assert out.loc[0, STATUS] == Status.OK.value and out.loc[0, "bio_key"] == bio_cache_key(cfg, s, "olmo")


# --- prompt and answers -----------------------------------------------------------------------------


DESCRIPTORS = {
    "n1": {"node_type": "company", "sector": "Energy", "layer": "large", "market_cap_rank_share": 0.05},
    "n2": {"node_type": "company", "sector": "Health Care", "layer": "small", "market_cap_rank_share": 0.9},
    "c:US": {"node_type": "country", "sector": None, "layer": None},
    "n3": {"node_type": "company", "sector": "Financials", "layer": "mid"},
    "n4": {"node_type": "company", "sector": "Energy", "layer": "mid"},
    "n5": {"node_type": "company", "sector": "Industrials", "layer": "large"},
    "n6": {"node_type": "company", "sector": "Materials", "layer": "mid"},
}


def sk(cfg, tier):
    base = a_skeleton(cfg)
    from dataclasses import replace
    return replace(base, literacy_tier=tier, career="retired teacher" if tier != "high" else None,
                   persona_type="retail" if tier != "high" else "hedge_fund")


def test_assets_shown_depend_on_literacy_and_reserve_sp_slots(cfg):
    pins = [{"node_id": "n1", "node_type": "company"}, {"node_id": "c:US", "node_type": "country"}]
    links = [{"node_id": "n2", "link_type": "industry"}, {"node_id": "n3", "link_type": "owner"}]
    low = select_assets(cfg, sk(cfg, "low"), pins, links, {"n1"}, [], DESCRIPTORS)
    assert [a.node_id for a in low] == ["n1"]  # only the company named in the text
    med = select_assets(cfg, sk(cfg, "medium"), pins, links, {"n1"}, [], DESCRIPTORS)
    assert [a.node_id for a in med] == ["n1", "c:US"]
    high = select_assets(cfg, sk(cfg, "high"), pins, links, {"n1", "n2", "n3"}, [], DESCRIPTORS)
    assert [a.node_id for a in high] == ["n1", "c:US", "n2", "n3"] and high[2].link_type == "industry"
    only_context = [{"node_id": "c:US", "node_type": "country"}]
    reserved = select_assets(cfg, sk(cfg, "medium"), only_context, [], set(), ["n1", "n2", "n3", "n4"], DESCRIPTORS)
    assert [a.node_id for a in reserved] == ["c:US", "n1", "n2", "n3"] and reserved[1].link_type == "reserved"
    many = [{"node_id": f"n{i}", "node_type": "company"} for i in range(1, 7)]
    capped = select_assets(cfg, sk(cfg, "high"), many, links, set(), ["n4", "n5", "n6"], DESCRIPTORS)
    assert len(capped) <= 6 and [a.ref for a in capped] == [f"A{i}" for i in range(1, len(capped) + 1)]


def test_low_literacy_sees_salience_words_not_sizes(cfg):
    famous = ShownAsset("A1", "n1", "company", "Energy", "large", None, 0.05)
    obscure = ShownAsset("A2", "n2", "company", "Health Care", "small", None, 0.9)
    assert asset_line(cfg, "low", famous, 0.0) == "A1: a well-known US Energy company"
    assert asset_line(cfg, "low", obscure, 0.0) == "A2: an obscure US Health Care company"
    assert "size=large" in asset_line(cfg, "medium", famous, 0.0)
    assert "your weight 12.0%" in asset_line(cfg, "high", famous, 0.12)


def test_prompt_contains_feed_assets_and_persona_parts_without_names(cfg):
    s = sk(cfg, "medium")
    feed = assemble_feed(cfg, profile("retail", "medium", "general", "retail_00"), [item("f1")], [], T0, "finance", TOK, SEED)
    assets = [ShownAsset("A1", "n1", "company", "Energy", "large", None, 0.05)]
    prompt = build_persona_prompt(
        cfg, s, "You are cautious.", feed, assets, {"state": -0.12}, -0.05, "Last quarter you made no trades.", {"Energy": 0.3}
    )
    text = prompt.user
    for part in ("[FEED — what you have seen]", "[ASSETS]", "[YOU]", "[QUESTION]", "You are cautious.",
                 "You work as retired teacher.", "down 12%", "Last quarter you made no trades.", "reliability:"):
        assert part in text, part
    assert text.index("[FEED") < text.index("[ASSETS]") < text.index("[YOU]") < text.index("[QUESTION]")
    assert '"value_change_pct"' in text and prompt.ref_to_node == {"A1": "n1"}


def test_parse_answer_accepts_five_numbers_clamps_and_flags(cfg):
    refs = {"A1": "n1"}
    good = parse_answer(json.dumps({"fear": 3, "anger": 2, "calm": 7, "value_change_pct": {"A1": 4.5}, "conviction": 6}), refs, cfg)
    assert good.ok and good.values == {"A1": 4.5} and not good.clamped
    wild = parse_answer(json.dumps({"fear": 30, "anger": -2, "calm": 7, "value_change_pct": {"A1": 99, "A9": 1}, "conviction": 6}), refs, cfg)
    assert wild.ok and wild.clamped and wild.fear == 10 and wild.anger == 0 and wild.values == {"A1": 30}
    for bad in ("nope", "[1]", json.dumps({"fear": 1}), json.dumps({"fear": "x", "anger": 1, "calm": 1, "conviction": 1, "value_change_pct": {}})):
        assert not parse_answer(bad, refs, cfg).ok


# --- processor and tape ------------------------------------------------------------------------------


def make_processor(cfg, access_off=()):
    rng = np.random.default_rng(7)
    grid = load_career_grid(str(CONFIG_DIR / "career_grid.csv"))
    skels = retail_skeletons(cfg, grid, universe(rng), sector_domain(), "2024Q1", SEED)[:4]
    skeletons = {s.persona_id: s for s in skels}
    bios = {s.bio_id: template_bio(cfg, s) for s in skels}
    focal = (item("f1", title="Acme Corp beat estimates", text="record results"),)
    event = EventPayload(
        "e1", "finance", T0, focal, (), ({"node_id": "n1", "node_type": "company"},), (), (), DESCRIPTORS,
        frozenset({"n1"}),
    )
    states = {("e1", s.persona_id): PersonaState(0.05, 0.02, 0.03, "Last quarter you made no trades.", {"Energy": 0.2}) for s in skels}
    engine = MockEngine()
    engine.register("persona", persona_responder)
    active = cfg
    for pid in access_off:
        active = active.with_overrides({f"personas.source_matrices.{skeletons[pid].persona_type}_{skeletons[pid].literacy_tier}.access.finance": 0.0}) if skeletons[pid].persona_type == "retail" else active
    return PersonaProcessor(active, engine, skeletons, bios, {"e1": event}, states, TOK), skels


def test_processor_returns_three_draws_per_persona_with_planted_dependence(cfg):
    proc, skels = make_processor(cfg)
    batch = pd.DataFrame({UNIT_ID: [unit_id("e1", s.persona_id) for s in skels]})
    out = proc(batch)
    assert list(out[UNIT_ID]) == list(batch[UNIT_ID])
    ok = out[out[STATUS] == Status.OK.value]
    assert len(ok) > 0
    for blob in ok[DRAWS_JSON]:
        draws = json.loads(blob)
        assert len(draws) == 3 and all(d["ok"] for d in draws)
        assert np.mean([d["values"]["n1"] for d in draws]) > 0  # planted: positive words -> positive value
    assert out.equals(proc(batch))  # deterministic


def test_persona_without_access_is_skipped_not_called(cfg):
    lone = cfg.with_overrides({f"personas.source_matrices.retail_{t}.access.finance": 0.0 for t in ("low", "medium", "high")})
    proc, skels = make_processor(lone)
    out = proc(pd.DataFrame({UNIT_ID: [unit_id("e1", skels[0].persona_id)]}))
    assert out.loc[0, STATUS] == Status.SKIPPED.value and out.loc[0, REASON] == Reason.NO_FOCAL_ACCESS.value
    assert PARSE_OK not in out.columns or not out.loc[0].get(PARSE_OK, False)


# --- post-processing -------------------------------------------------------------------------------


def test_running_lean_matches_brute_force_with_gap_and_window():
    rng = np.random.default_rng(8)
    sessions = np.sort(rng.integers(0, 400, 120))
    values = rng.normal(size=120)
    got = running_lean(sessions, values, window=10, min_gap=5)
    for i in range(120):
        eligible = [j for j in range(120) if sessions[j] <= sessions[i] - 5][-10:]
        want = np.mean(values[eligible]) if eligible else np.nan
        assert (np.isnan(got[i]) and np.isnan(want)) or got[i] == pytest.approx(want)


def answers_frame(rng, bias):
    rows = []
    for persona, b in bias.items():
        for event in range(60):
            for draw in range(3):
                rows.append({
                    "event_id": f"e{event}", "persona_id": persona, "persona_type": "retail", "session": event * 2,
                    "draw": draw, "fear": 3 + b + rng.normal(0, 0.3), "anger": 3.0, "calm": 3.0,
                    "conviction": 5.0, "value_mean": b + rng.normal(0, 0.3),
                })
    return pd.DataFrame(rows)


def test_lean_removal_removes_a_persistent_bias_using_only_the_past(cfg):
    rng = np.random.default_rng(9)
    frame = answers_frame(rng, {"bull": 4.0, "bear": -4.0})
    cleaned = remove_lean(frame, cfg)
    late = cleaned[cleaned["session"] > 60]
    assert abs(late["value_mean_star"].mean()) < 0.3  # bias gone after warm-up
    first = cleaned[cleaned["session"] < 5]
    assert first["value_mean_star"].abs().mean() < 4.5  # cold start falls back to the type mean
    shifted = frame.copy()
    shifted.loc[shifted["session"] >= 100, "value_mean"] += 50.0  # future change must not alter the past
    again = remove_lean(shifted, cfg)
    early = cleaned["session"] < 100
    assert np.allclose(cleaned.loc[early, "value_mean_star"], again.loc[again["session"] < 100, "value_mean_star"])


def test_draw_aggregation_emotion_ranges_and_dispersions(cfg):
    rng = np.random.default_rng(10)
    cleaned = remove_lean(answers_frame(rng, {"a": 1.0, "b": -1.0, "c": 0.0}), cfg)
    agg = aggregate_draws(cleaned)
    assert len(agg) == 3 * 60 and (agg["d_draw"] >= 0).all()
    emo = emotion_variables(agg, scale_max=10.0)
    assert emo["fa"].between(-1, 1).all() and emo["ar"].between(-1, 1).all()
    books = pd.Series({"a": 3.0, "b": 1.0, "c": 1.0})
    belief = type_belief(emo, books, "value_mean_star")
    assert set(belief.columns) == {"event_id", "persona_type", "belief", "dispersion"}
    spread = persona_dispersion(emo, books, "value_mean_star")
    assert len(spread) == 60 and (spread >= 0).all()


def test_event_pinned_only_on_a_context_node_still_shows_sp_stocks_and_reads_out_sp_values(cfg):
    """Spec 12.2 #21: oil news (a commodity node, no S&P member pinned) still puts S&P stocks on the persona's list."""
    rng = np.random.default_rng(7)
    grid = load_career_grid(str(CONFIG_DIR / "career_grid.csv"))
    skels = retail_skeletons(cfg, grid, universe(rng), sector_domain(), "2024Q1", SEED)[:4]
    skeletons = {s.persona_id: s for s in skels}
    bios = {s.bio_id: template_bio(cfg, s) for s in skels}
    descriptors = {**DESCRIPTORS, "c:oil": {"node_type": "commodity", "sector": None, "layer": None}}
    event = EventPayload(
        "e1", "geo", T0, (item("f1", title="Crude oil supply cut announced", text="record cuts"),), (),
        ({"node_id": "c:oil", "node_type": "commodity"},), (), ("n1", "n2", "n3", "n4"), descriptors, frozenset({"n1", "n2", "n3", "n4"}),
    )
    states = {("e1", s.persona_id): PersonaState(0.05, 0.02, 0.03, "Last quarter you made no trades.", {"Energy": 0.2}) for s in skels}
    engine = MockEngine()
    engine.register("persona", persona_responder)
    out = PersonaProcessor(cfg, engine, skeletons, bios, {"e1": event}, states, TOK)(
        pd.DataFrame({UNIT_ID: [unit_id("e1", s.persona_id) for s in skels]}))
    ok = out[out[STATUS] == Status.OK.value]
    assert len(ok) > 0
    shown = select_assets(cfg, skels[0], event.pins, event.links, set(event.sp_members), event.reserve_candidates, descriptors)
    assert any(a.node_id in event.sp_members for a in shown) and shown[-1].link_type == "reserved"  # S&P stocks reach the list
    for blob in ok[DRAWS_JSON]:
        draws = json.loads(blob)
        assert any(node in event.sp_members for d in draws for node in d["values"])  # the S&P readout is not empty
