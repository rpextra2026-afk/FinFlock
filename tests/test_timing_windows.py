"""Day-0 timing rule, calendar closures, windows, embargo and cutoff splits (spec 12.2 #2, #19)."""

from __future__ import annotations

import pandas as pd
import pytest

from oryxis.common.timing import TradingCalendar, to_utc
from oryxis.common.windows import (
    DEV,
    EMBARGO,
    IN_CUTOFF,
    POST_CUTOFF,
    TEST,
    WindowError,
    bucket_windows,
    clean_start,
    cutoff_part,
)

ET = "America/New_York"


@pytest.fixture(scope="module")
def cal():
    return TradingCalendar("XNYS", "2012-01-01", "2025-12-31")


def day0(cal, stamp: str) -> str:
    utc = to_utc([stamp], ET)
    return str(cal.day0(utc)[0].date())


def test_special_closures_are_not_sessions(cal):
    for closed in ["2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"]:
        assert not cal.is_session(closed), closed
    assert cal.is_session("2025-01-10")


def test_before_close_is_same_day(cal):
    assert day0(cal, "2024-03-12 10:00:00") == "2024-03-12"
    assert day0(cal, "2024-03-12 15:59:59") == "2024-03-12"


def test_exactly_at_close_and_after_close_is_next_session(cal):
    assert day0(cal, "2024-03-12 16:00:00") == "2024-03-13"
    assert day0(cal, "2024-03-12 21:00:00") == "2024-03-13"


def test_friday_after_close_weekend_and_holiday_roll_forward(cal):
    assert day0(cal, "2024-03-15 17:00:00") == "2024-03-18"  # Friday after close -> Monday
    assert day0(cal, "2024-03-16 11:00:00") == "2024-03-18"  # Saturday
    assert day0(cal, "2024-07-04 11:00:00") == "2024-07-05"  # holiday


def test_special_closure_days_roll_to_next_session(cal):
    assert day0(cal, "2025-01-09 10:00:00") == "2025-01-10"
    assert day0(cal, "2018-12-05 10:00:00") == "2018-12-06"


def test_early_close_day_uses_the_early_close(cal):
    # 2024-11-29 (day after Thanksgiving) closes at 13:00 ET.
    assert day0(cal, "2024-11-29 12:59:00") == "2024-11-29"
    assert day0(cal, "2024-11-29 13:30:00") == "2024-12-02"


def test_dst_change_does_not_shift_the_close(cal):
    # 2024-03-11 is the first Monday after the spring-forward; close is 16:00 EDT = 20:00 UTC.
    assert str(cal.day0(pd.DatetimeIndex(["2024-03-11 19:59:00"], tz="UTC"))[0].date()) == "2024-03-11"
    assert str(cal.day0(pd.DatetimeIndex(["2024-03-11 20:00:00"], tz="UTC"))[0].date()) == "2024-03-12"


def test_date_only_items_use_that_date_or_the_next_session(cal):
    out = cal.day0_date_only(["2024-03-12", "2024-03-16", "2024-07-04"])
    assert [str(d.date()) for d in out] == ["2024-03-12", "2024-03-18", "2024-07-05"]


def test_day0_never_precedes_availability_property(cal):
    rng = pd.date_range("2020-01-01", "2024-12-31", freq="7h13min", tz="UTC")
    sessions = cal.day0(rng)
    closes = cal.closes_utc[cal.sessions.get_indexer(sessions)]
    assert (closes > rng).all()
    previous = cal.closes_utc[cal.sessions.get_indexer(sessions) - 1]
    assert (previous <= rng).all()


def test_shift_and_between(cal):
    assert str(cal.shift("2024-03-15", 1).date()) == "2024-03-18"
    assert str(cal.shift("2024-03-18", -1).date()) == "2024-03-15"
    assert cal.count_between("2023-12-31", "2024-01-09") == 5


def test_every_bucket_has_a_valid_embargo_and_stays_inside_the_data(cfg, cal):
    for bucket in ("db1", "tessera", "db3"):
        w = bucket_windows(cfg, bucket, cal)
        assert cal.count_between(w.dev_end, w.test_start) >= w.embargo_sessions
        assert w.test_end <= pd.Timestamp(cfg.get("windows.data_end"))


def test_c10_dates(cfg, cal):
    w1, w3 = bucket_windows(cfg, "db1", cal), bucket_windows(cfg, "db3", cal)
    assert (str(w1.dev_end.date()), str(w1.test_start.date())) == ("2021-12-31", "2022-01-10")
    assert (str(w3.dev_end.date()), str(w3.test_start.date())) == ("2018-12-31", "2019-01-09")


def test_short_embargo_and_late_test_end_are_refused(cfg, cal):
    short = cfg.with_overrides({"windows.buckets.tessera.test_start": "2024-01-03"})
    with pytest.raises(WindowError):
        bucket_windows(short, "tessera", cal)
    late = cfg.with_overrides({"windows.buckets.tessera.test_end": "2026-03-31"})
    with pytest.raises(WindowError):
        bucket_windows(late, "tessera", cal)


def test_window_labels(cfg, cal):
    w = bucket_windows(cfg, "tessera", cal)
    labels = w.label(pd.DatetimeIndex(["2023-12-29", "2024-01-03", "2024-01-09", "2026-01-02"]))
    assert list(labels) == [DEV, EMBARGO, TEST, "outside"]


def test_clean_start_is_first_session_of_month_after_cutoff(cfg, cal):
    models = cfg.get("llms.models")
    assert str(clean_start(models["llm3"]["stated_cutoff"], cal).date()) == "2024-01-02"
    assert str(clean_start(models["llm2"]["stated_cutoff"], cal).date()) == "2025-01-02"
    assert str(clean_start(models["llm1"]["stated_cutoff"], cal).date()) == "2025-02-03"
    assert str(clean_start("2024-12", cal, empirical_cutoff="2025-03").date()) == "2025-04-01"


def test_test_part_split_per_llm(cfg, cal):
    w = bucket_windows(cfg, "tessera", cal)
    dates = pd.DatetimeIndex(["2023-06-01", "2024-06-03", "2025-01-02", "2025-06-02"])
    olmo = cutoff_part(dates, w, clean_start("2024-12", cal))
    assert list(olmo) == [None, IN_CUTOFF, POST_CUTOFF, POST_CUTOFF]
    llama = cutoff_part(dates, w, clean_start("2023-12", cal))
    assert list(llama) == [None, POST_CUTOFF, POST_CUTOFF, POST_CUTOFF]
    # replication buckets lie wholly before every cutoff: no post_cutoff days at all
    w1 = bucket_windows(cfg, "db1", cal)
    inside = pd.DatetimeIndex(["2022-06-01", "2023-06-01"])
    assert set(cutoff_part(inside, w1, clean_start("2023-12", cal))) == {IN_CUTOFF}
