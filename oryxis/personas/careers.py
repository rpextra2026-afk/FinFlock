"""The career grid (spec 6.5): a career fixes exactly literacy tier, domain and attention."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from oryxis.common.config import Config

CAREER_ID = "career_id"
CAREER = "career"
LITERACY = "literacy"
DOMAIN = "domain"


@dataclass(frozen=True)
class Career:
    career_id: int
    name: str
    literacy: str
    domain: str


def load_career_grid(path: str) -> list[Career]:
    frame = pd.read_csv(path)
    return [
        Career(int(r[CAREER_ID]), str(r[CAREER]), str(r[LITERACY]), str(r[DOMAIN]))
        for _, r in frame.iterrows()
    ]


def retail_career_sequence(grid: list[Career], cfg: Config) -> list[Career]:
    """Careers of the retail personas: every career once, then the repeated common ones (spec 6.5)."""
    by_id = {c.career_id: c for c in grid}
    extras = [by_id[i] for i in cfg.get("personas.retail.extra_careers")]
    return list(grid) + extras


def literacy_value(cfg: Config, tier: str) -> float:
    return float(cfg.get(f"personas.literacy_values.{tier}"))
