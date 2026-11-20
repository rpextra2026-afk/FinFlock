"""The single results file (experiment configuration): tables, figures, diagnostics and lineage in ONE bundle.

The report states numbers and comparison values, never verdicts. ``assert_no_verdict`` scans the static text
of every template this module emits; table cells hold numbers produced by code from parquet files, each
table carries a lineage id (config hash, data hash, commit, seed), and mock or synthetic inputs are refused.
"""

from __future__ import annotations

import html
import io
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from finfm.common.config import Config  # noqa: E402
from finfm.common.hashing import canonical_json, stable_hash  # noqa: E402
from finfm.compute.bundle import PartSource, write_bundle  # noqa: E402

FIGURE_DATA_SUFFIX = ".csv"
LINEAGE_ID_LENGTH = 16  # characters of the hash shown as the lineage id


class ReportError(RuntimeError):
    """Raised when a report would break a rule (verdict wording, mock or synthetic inputs)."""


@dataclass
class TableSpec:
    key: str  # T8, T9 ...
    title: str
    frame: pd.DataFrame
    note: str = ""
    sources: tuple[str, ...] = ()


@dataclass
class FigureSpec:
    key: str
    title: str
    data: pd.DataFrame  # the numbers behind the picture, saved beside it
    kind: str = "line"  # line | bar | scatter
    x: str | None = None
    ylabel: str = ""


@dataclass
class Lineage:
    config_hash: str
    data_hash: str
    git_commit: str | None
    seed: int
    engine: str
    synthetic: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return stable_hash([self.config_hash, self.data_hash, self.git_commit, self.seed, self.engine])[:LINEAGE_ID_LENGTH]


def assert_no_verdict(text: str, cfg: Config) -> None:
    """Static report text must not contain verdict vocabulary (R8)."""
    words = {w.lower() for w in cfg.get("eval.report.forbidden_words")}
    found = sorted({w for w in re.findall(r"[A-Za-z]+", text) if w.lower() in words})
    if found:
        raise ReportError(f"verdict wording in report text: {found}")


def check_inputs(lineage: Lineage) -> None:
    if lineage.synthetic or lineage.engine == "mock":
        raise ReportError("refusing to build a results report from synthetic or mock inputs")


def latex(frame: pd.DataFrame, decimals: int) -> str:
    """LaTeX tabular of a table (numbers rounded; text escaped)."""
    cols = "l" * frame.shape[1]
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")  # noqa: E731
    rows = [" & ".join(esc(c) for c in frame.columns) + r" \\", r"\hline"]
    for _, row in frame.iterrows():
        rows.append(" & ".join(esc(f"{v:.{decimals}f}") if isinstance(v, float) else esc(v) for v in row) + r" \\")
    return "\\begin{tabular}{" + cols + "}\n" + "\n".join(rows) + "\n\\end{tabular}\n"


def plot(spec: FigureSpec, cfg: Config) -> tuple[bytes, bytes]:
    """SVG and PNG bytes of a figure; the data frame is what the picture shows."""
    colours = list(cfg.get("eval.report.palette"))
    look = cfg.get("eval.report.figure")
    fig, ax = plt.subplots(figsize=tuple(look["size"]))
    data = spec.data
    xs = data[spec.x] if spec.x else data.index
    value_cols = [c for c in data.columns if c != spec.x]
    if spec.kind == "bar":
        width = look["bar_group_width"] / len(value_cols)
        for i, c in enumerate(value_cols):
            ax.bar([j + i * width for j in range(len(xs))], data[c], width=width, color=colours[i % len(colours)], label=str(c))
        centre = (look["bar_group_width"] - width) / 2
        ax.set_xticks([j + centre for j in range(len(xs))], [str(x) for x in xs], rotation=look["tick_rotation"], ha="right")
    elif spec.kind == "scatter":
        ax.scatter(data[value_cols[0]], data[value_cols[1]], color=colours[0], s=look["scatter_size"])
        ax.set_xlabel(str(value_cols[0]))
    else:
        for i, c in enumerate(value_cols):
            ax.plot(xs, data[c], color=colours[i % len(colours)], label=str(c), linewidth=look["line_width"])
    ax.set_title(spec.title, loc="left", fontsize=look["title_size"])
    ax.set_ylabel(spec.ylabel)
    ax.grid(alpha=look["grid_alpha"])
    if len(value_cols) > 1 and spec.kind != "scatter":
        ax.legend(frameon=False, fontsize=look["legend_size"])
    fig.tight_layout()
    out = {}
    for fmt in ("svg", "png"):
        buffer = io.BytesIO()
        fig.savefig(buffer, format=fmt, dpi=int(cfg.get("eval.report.figure_dpi")))
        out[fmt] = buffer.getvalue()
    plt.close(fig)
    return out["svg"], out["png"]


def table_html(spec: TableSpec, decimals: int, lineage_id: str) -> str:
    body = spec.frame.round(decimals).to_html(index=False, border=0, classes="t", na_rep="")
    note = f"<p class='n'>{html.escape(spec.note)}</p>" if spec.note else ""
    return f"<section id='{spec.key}'><h2>{spec.key}. {html.escape(spec.title)}</h2>{note}{body}<p class='l'>lineage {lineage_id}</p></section>"


STYLE = (
    "body{font:14px/1.45 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}"
    "table.t{border-collapse:collapse;font-size:12px}table.t th,table.t td{padding:3px 9px;border-bottom:1px solid #ddd;text-align:right}"
    "table.t th:first-child,table.t td:first-child{text-align:left}h1{font-size:20px}h2{font-size:15px;margin-top:2rem}"
    ".n,.l{color:#555;font-size:12px}figure{margin:1rem 0}"
)
STATIC_TEXT = {
    "heading": "Results",
    "intro": "Every number below is produced by code from the files in this bundle; each table carries a lineage id. "
             "Intervals and comparison values stand beside every estimate.",
    "lineage": "Lineage",
    "contents": "Contents",
}


def build_report(
    tables: Sequence[TableSpec], figures: Sequence[FigureSpec], diagnostics: Mapping[str, Any], lineage: Lineage, cfg: Config,
) -> tuple[str, dict[str, bytes]]:
    """The HTML page (figures embedded as SVG) and the member files of the bundle."""
    check_inputs(lineage)
    assert_no_verdict(" ".join(STATIC_TEXT.values()), cfg)
    decimals = int(cfg.get("eval.report.decimals"))
    files: dict[str, bytes] = {}
    sections = []
    for spec in tables:
        assert_no_verdict(f"{spec.title} {spec.note} {' '.join(map(str, spec.frame.columns))}", cfg)
        files[f"tables/{spec.key}.csv"] = spec.frame.to_csv(index=False).encode("utf-8")
        files[f"tables/{spec.key}.tex"] = latex(spec.frame, decimals).encode("utf-8")
        sections.append(table_html(spec, decimals, lineage.id))
    for spec in figures:
        assert_no_verdict(spec.title, cfg)
        svg, png = plot(spec, cfg)
        files[f"figures/{spec.key}.svg"], files[f"figures/{spec.key}.png"] = svg, png
        files[f"figures/{spec.key}{FIGURE_DATA_SUFFIX}"] = spec.data.to_csv().encode("utf-8")
        sections.append(f"<figure id='{spec.key}'>{svg.decode('utf-8')}<figcaption>{html.escape(spec.title)}</figcaption></figure>")
    files["diagnostics/diagnostics.json"] = canonical_json(dict(diagnostics)).encode("utf-8")
    files["lineage/lineage.json"] = canonical_json({**lineage.__dict__, "id": lineage.id}).encode("utf-8")
    toc = "".join(f"<li><a href='#{s.key}'>{s.key}. {html.escape(s.title)}</a></li>" for s in tables)
    page = (
        f"<!doctype html><meta charset='utf-8'><title>{STATIC_TEXT['heading']}</title><style>{STYLE}</style>"
        f"<h1>{STATIC_TEXT['heading']}</h1><p>{STATIC_TEXT['intro']}</p><h2>{STATIC_TEXT['contents']}</h2><ul>{toc}</ul>"
        f"{''.join(sections)}<h2>{STATIC_TEXT['lineage']}</h2><pre>{html.escape(files['lineage/lineage.json'].decode())}</pre>"
    )
    files["REPORT.html"] = page.encode("utf-8")
    return page, files


def write_results_bundle(
    path: Path, files: Mapping[str, bytes], data_parts: Sequence[PartSource], lineage: Lineage, cfg: Config,
) -> dict[str, Any]:
    """Write the one results file: report, tables, figures, diagnostics, lineage and the raw data parts."""
    return write_bundle(
        path, job_id="results", shard_id="all", seq=1, parents=[], parts=data_parts, files=files,
        meta={"lineage": lineage.id, "config_hash": lineage.config_hash},
        compress_level=int(cfg.get("runtime.bundle_compression_level")),
    )
