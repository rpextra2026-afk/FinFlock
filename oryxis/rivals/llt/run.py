"""R3, basic: Lopez-Lira and Tang. One LLM call per qualifying single-firm finance headline.

The prompt is verbatim from the paper (single user message, no system prompt). The model sees the real
company name and the real headline, exactly as the published method does; the masked sensitivity is a
separate run. An unparseable answer is a counted failure with score 0.
"""

from __future__ import annotations

import pandas as pd

from oryxis.common.config import Config
from oryxis.common.schema import DETAIL, PARSE_OK, REASON, STATUS, TOKENS_IN, TOKENS_OUT, UNIT_ID, Reason, Status
from oryxis.compute.engine import Engine, GenParams, GenRequest

TASK = "llt"
COMPANY = "company_name"
HEADLINE = "headline"
ANSWER = "answer"
SCORE = "score"
RAW = "raw_output"
KINDS_FINANCE = "finance"


def select_events(
    events: pd.DataFrame, pins: dict[str, list[str]], names: dict[str, str], is_member, cfg: Config
) -> pd.DataFrame:
    """Finance events with a title that name exactly one S&P 500 member (the published method scores one firm)."""
    rows = []
    for row in events.itertuples():
        if row.kind != KINDS_FINANCE or not str(row.title).strip():
            continue
        members = [n for n in pins.get(row.event_id, []) if is_member(n, row.session_day0)]
        if len(members) == 1 and members[0] in names:
            rows.append({UNIT_ID: row.event_id, "security_id": members[0], COMPANY: names[members[0]],
                         HEADLINE: row.title, "decision_date": row.decision_date})
    return pd.DataFrame(rows, columns=[UNIT_ID, "security_id", COMPANY, HEADLINE, "decision_date"])


def parse_answer(text: str, cfg: Config) -> tuple[str | None, int]:
    """First line mapped YES / NO / UNKNOWN to +1 / -1 / 0; anything else is a failure (score 0)."""
    first = text.strip().splitlines()[0].strip().strip('"\'.:').upper() if text.strip() else ""
    mapping = cfg.get("rivals.llt.answers")
    return (first, int(mapping[first])) if first in mapping else (None, 0)


class LLTProcessor:
    def __init__(self, cfg: Config, engine: Engine) -> None:
        self.cfg, self.engine = cfg, engine
        spec = cfg.get("rivals.llt")
        self.params = GenParams(temperature=float(spec["temperature"]), top_p=1.0, top_k=None,
                                max_tokens=int(spec["max_tokens"]), seed=int(cfg.get("runtime.base_seed")))

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        template = str(self.cfg.get("prompts.llt.user"))
        requests = [GenRequest(str(r[UNIT_ID]), template.format(company_name=r[COMPANY], headline=r[HEADLINE]), task=TASK)
                    for _, r in batch.iterrows()]
        rows = []
        for result in self.engine.generate(requests, self.params):
            answer, score = parse_answer(result.texts[0], self.cfg)
            rows.append({UNIT_ID: result.unit_id, STATUS: Status.OK.value, REASON: None if answer else Reason.PARSE_FAILED.value,
                         DETAIL: None, PARSE_OK: answer is not None, ANSWER: answer, SCORE: score, RAW: result.texts[0],
                         TOKENS_IN: result.tokens_in, TOKENS_OUT: result.tokens_out})
        return pd.DataFrame(rows)
