"""One MASS agent decision: investing style, stock pool selector, then the choice (repository logic).

The prompts are the repository's own (``config/mass_prompts.yaml``). With the model at temperature 0 a
retry returns the same text, so retries are skipped; the repository calls the model at temperature 0.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from finfm.common.config import Config

SELECTORS = ("RandomStockSelector", "IndustryEqualStockSelector", "MVEqualStockSelector")
MV_GROUPS = 5
PERCENT = 100
NEWS_EXAMPLES = 3
TABLE_LAYOUT_REPO = "repo"


def data_input(descriptions: Mapping[str, str], news_titles: list[str] | None) -> str:
    """The "Investors read following information ..." sentence of the repository."""
    headers = dict(descriptions)
    if news_titles is not None:
        headers["news"] = f"Investors read news to make investment descisions. The example news is: {news_titles[:NEWS_EXAMPLES]}"
    return (
        "Investors read following information to make investment decisions. The information is in key-value "
        f"format, key representing name, value representing descriptions.: {headers}. \n"
    )


def style_prompt(cfg: Config, descriptions: Mapping[str, str], news_titles: list[str] | None, macro: str | None) -> str:
    prompts = cfg.get("mass_prompts.prompts")
    base = data_input(descriptions, news_titles)
    if cfg.get("rivals.mass.use_macro_data") and macro:
        return prompts["investing_style_instruction_with_marco_data"].format(
            examples=prompts["investing_style_with_macro_data_example"], input_data=base, macro_data=macro)
    return prompts["investing_style_instruction"].format(examples=prompts["invseting_style_example"], input_data=base)


def parse_style(text: str) -> tuple[dict | None, str]:
    """Style JSON and the selector name (anything unknown falls back to random, as the repository does)."""
    try:
        style = json.loads(text)
        selector = style["Details"]["StockPoolSelector"]
    except (ValueError, KeyError, TypeError):
        return None, SELECTORS[0]
    return style, selector if isinstance(selector, str) and selector in SELECTORS else SELECTORS[0]


def select_pool(selector: str, pool: pd.DataFrame, n: int, rng: np.random.Generator) -> list[str]:
    """Stocks shown to the agent: ``random.choices`` (with replacement) with the selector's weights."""
    stocks = pool["Stock"].tolist()
    if selector == "IndustryEqualStockSelector":
        sizes = pool.groupby("Industry")["Stock"].transform("count")
        weights = (1.0 / sizes).to_numpy()
    elif selector == "MVEqualStockSelector":
        rank = pool["FREE_MV"].rank(ascending=False)
        group = ((rank - 1) // (len(pool) / MV_GROUPS)).astype(int)
        weights = (1.0 / group.map(group.value_counts())).to_numpy()
    else:
        weights = np.ones(len(stocks))
    picks = rng.choice(len(stocks), size=n, replace=True, p=weights / weights.sum())
    return [stocks[i] for i in picks]


def decision_prompt(cfg: Config, descriptions: Mapping[str, str], tables: Mapping[str, pd.DataFrame], style: Mapping, n_pick: int) -> str:
    """The repository's decision prompt (basic_agent.py): the instruction with the few-shot example (``rivals.mass.decision_example``), then
    ``Input Data for investing decision:1. descriptions: {str(descriptions)}, 2. input data: {str(tables)}.`` and `` Investment strategy: {str(style)}``.
    ``str(tables)`` is the dict of DataFrames as pandas prints it (text tables, row index kept). The repository sets no pandas display options, so what
    the model saw there depended on the terminal; ``rivals.mass.table_display`` pins them here so that no row and no column is hidden."""
    prompts = cfg.get("mass_prompts.prompts")
    spec = cfg.get("rivals.mass")
    if str(spec["table_layout"]) != TABLE_LAYOUT_REPO:
        raise ValueError(f"unknown rivals.mass.table_layout {spec['table_layout']!r}")
    options = [item for key, value in spec["table_display"].items() for item in (key, value)]
    with pd.option_context(*options):
        data = str(dict(tables))
    body = f"Input Data for investing decision:1. descriptions: {str(dict(descriptions))}, 2. input data: {data}."
    return prompts["investing_decision_instruction"].format(
        num_stocks=n_pick, examples=prompts[str(spec["decision_example"])], input_data=body + f" Investment strategy: {str(dict(style))}")


def parse_decision(text: str, shown: list[str]) -> list[str] | None:
    """Chosen stocks from ``{"Stock": [...]}``; an illegal code (not shown) invalidates the decision."""
    try:
        chosen = json.loads(text)["Stock"]
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(chosen, list) or any(str(s) not in set(shown) for s in chosen):
        return None
    return [str(s) for s in chosen]


def macro_text(series: Mapping[str, float], quantile: float) -> str:
    """US analogue of the repository's macro sentence (policy rate, CPI, ten-year yield, valuation percentile, market return)."""
    return (
        f"The latest 3-month Treasury bill rate is {series['tbill']}. "
        f"The latest month US CPI YOY growth rate is {series['cpi_yoy']}. "
        f"The latest yield of US ten year government bonds is {series['ten_year']}%, while yield increases "
        f"{round(series['d1'] * PERCENT)} BP over past one day, increases {round(series['d30'] * PERCENT)} BP over past one month, "
        f"increases {round(series['d180'] * PERCENT)} BP over past half an year. "
        f"The latest Shiller CAPE is {series['cape']}, and the current CAPE of the S&P 500 is at the {quantile} percentile "
        "over the past 5 years(0 indicates most undervalued, and 100 indicates most overvalued). "
        f"The latest market sentiement index got {round(series['market_return'] * PERCENT, 2)} % return."
    )
