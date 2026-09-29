"""drift_alarm: anytime-valid e-process + Page-Hinkley diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.drift_alarm import (
    EProcessDriftAlarm,
    PageHinkleyAlarm,
    monitor_stream,
)


def test_stationary_short_stream_no_alarm() -> None:
    """Gentle bet (lam=0.2): log-e caps at 50*log(1.2)~9 < log(20)=3.0?
    No — but the martingale cannot reach 3.0 on a 50-step iid draw without
    an implausibly one-sided run; assert zero alarms across 30 seeds."""
    for s in range(30):
        rng = np.random.default_rng(s)
        ep = EProcessDriftAlarm(alpha=0.05, lam=0.2)
        for x in rng.normal(0.0, 1.0, 50):
            ep.update(float(x))
        assert not ep.alarmed, f"seed {s} false-alarmed"


def test_persistent_mean_from_start_no_alarm() -> None:
    """Level-shift semantics: a mean that exists from step 1 is absorbed
    into the running baseline — no 'change' to alarm on."""
    rng = np.random.default_rng(0)
    ep = EProcessDriftAlarm(alpha=0.05, burn_in=10)
    last = None
    for x in rng.normal(0.6, 1.0, 400):
        last = ep.update(float(x))
    assert not ep.alarmed
    assert last is not None and last.statistic < 1.0


def test_drift_midstream_alarms_after_change() -> None:
    rng = np.random.default_rng(1)
    ep = EProcessDriftAlarm(alpha=0.05, burn_in=10)
    for x in rng.normal(0.0, 1.0, 200):
        ep.update(float(x))
    assert not ep.alarmed
    for x in rng.normal(0.8, 1.0, 200):
        ep.update(float(x))
    assert ep.alarmed


def test_negative_drift_shrinks_e() -> None:
    rng = np.random.default_rng(2)
    ep = EProcessDriftAlarm(alpha=0.05, burn_in=10)
    last = None
    for x in rng.normal(-0.6, 1.0, 200):
        last = ep.update(float(x))
    assert not ep.alarmed
    assert last is not None and last.statistic < 1.0


def test_nonfinite_is_inconclusive_not_evidence() -> None:
    ep = EProcessDriftAlarm(alpha=0.05, burn_in=5)
    rng = np.random.default_rng(0)
    for x in rng.normal(0.0, 1.0, 20):
        ep.update(float(x))
    step = ep.update(float("nan"))
    assert step.inconclusive
    step2 = ep.update(float("inf"))
    assert step2.inconclusive
    assert not ep.alarmed


def test_page_hinkley_alarms_on_shift() -> None:
    rng = np.random.default_rng(3)
    ph = PageHinkleyAlarm(burn_in=10)
    for x in rng.normal(0.0, 1.0, 100):
        ph.update(float(x))
    for x in rng.normal(1.5, 1.0, 100):
        ph.update(float(x))
    assert ph.alarmed


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"lam": 0.0},
        {"lam": 1.5},
        {"init_scale": -1.0},
        {"burn_in": 0},
    ],
)
def test_fails_closed_args(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        EProcessDriftAlarm(**kwargs)  # type: ignore[arg-type]


def test_monitor_stream_report_shape() -> None:
    rng = np.random.default_rng(0)
    stream = list(rng.normal(0.0, 1.0, 100)) + list(rng.normal(1.0, 1.0, 150))
    rep = monitor_stream(stream, alpha=0.05)
    assert rep["kind"] == "drift_alarm.v1"
    assert rep["research_only"] is True
    ep = rep["eprocess"]
    assert isinstance(ep, dict)
    assert ep["alarmed"] is True
    assert ep["final_evalue"] >= 20.0  # 1/alpha
    assert "page_hinkley" in rep


def test_mc_false_alarm_rate() -> None:
    """Validity: P(ever alarm | null) <= alpha. Across 25 x 300-step streams
    the alarm count stays inside the ~5% bound (expect ~1, allow <=3)."""
    alarms = 0
    for s in range(25):
        rng = np.random.default_rng(1000 + s)
        ep = EProcessDriftAlarm(alpha=0.05, burn_in=8)
        for x in rng.normal(0.0, 1.0, 200):
            ep.update(float(x))
        alarms += int(ep.alarmed)
    assert alarms <= 3  # alpha=0.05 → expect ~1; allow slack
