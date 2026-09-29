"""conformal_monitor: Vovk conformal martingale over sliding windows."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.conformal_monitor import (
    CONFORMAL_MONITOR_SCHEMA,
    ConformalMartingale,
    monitor,
)


def test_iid_stream_no_alarm_short_run() -> None:
    for s in range(20):
        rng = np.random.default_rng(s)
        cm = ConformalMartingale(alpha=0.05, window=40)
        for x in rng.normal(0.0, 1.0, 120):
            cm.update(float(x))
        assert not cm.alarmed, f"seed {s} false-alarmed"


def test_variance_shift_alarms() -> None:
    """σ triples mid-stream — a level-shift detector is blind to this
    (mean unchanged); the conformal martingale catches it."""
    rng = np.random.default_rng(0)
    cm = ConformalMartingale(alpha=0.05, window=40)
    for x in rng.normal(0.0, 1.0, 120):
        cm.update(float(x))
    for x in rng.normal(0.0, 3.0, 200):
        cm.update(float(x))
    assert cm.alarmed
    assert (cm.alarm_index or 0) >= 40


def test_mean_shift_also_alarms() -> None:
    rng = np.random.default_rng(1)
    cm = ConformalMartingale(alpha=0.05, window=30)
    for x in rng.normal(0.0, 1.0, 100):
        cm.update(float(x))
    for x in rng.normal(2.0, 1.0, 150):
        cm.update(float(x))
    assert cm.alarmed


def test_stationary_after_warmup_martingale_bounded() -> None:
    rng = np.random.default_rng(2)
    cm = ConformalMartingale(alpha=0.05, window=30)
    last = 1.0
    for x in rng.normal(0.0, 1.0, 200):
        _, last = cm.update(float(x))
    assert last < 20.0  # never approached 1/alpha


def test_nonfinite_consumes_step_inconclusive() -> None:
    cm = ConformalMartingale(window=20)
    rng = np.random.default_rng(0)
    for x in rng.normal(0.0, 1.0, 30):
        cm.update(float(x))
    p, _m = cm.update(float("nan"))
    assert np.isnan(p)
    assert cm._n_inconclusive == 1
    # the nan was NOT added to the calibration bag or xs stream
    assert all(np.isfinite(v) for v in cm._xs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"kappa": 0.0},
        {"kappa": 1.0},
        {"window": 3},
        {"init_scale": -1.0},
    ],
)
def test_fails_closed_args(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ConformalMartingale(**kwargs)  # type: ignore[arg-type]


def test_monitor_report_shape() -> None:
    rng = np.random.default_rng(0)
    rep = monitor(
        list(rng.normal(0.0, 1.0, 80)) + list(rng.normal(0.0, 4.0, 150)),
        alpha=0.05,
    )
    assert rep["kind"] == CONFORMAL_MONITOR_SCHEMA
    assert rep["research_only"] is True
    assert rep["live_pnl_claim"] is False
    assert rep["alarmed"] is True
    assert rep["final_martingale"] >= 20.0
    assert rep["n_obs"] == 230


def test_conformal_p_bounded() -> None:
    cm = ConformalMartingale(window=25)
    rng = np.random.default_rng(4)
    for x in rng.normal(0.0, 1.0, 60):
        p, _ = cm.update(float(x))
        assert 0.0 < p <= 1.0


def test_fixed_vs_adaptive_mode_semantics() -> None:
    """Same stream: fixed bag alarms on the persistent regime change;
    adaptive re-conforms after the window drains (episode detector)."""
    rng = np.random.default_rng(0)
    prefix = list(rng.normal(0.0, 1.0, 60))
    shifted = list(rng.normal(0.0, 4.0, 300))
    cm_fixed = ConformalMartingale(alpha=0.05, window=40, mode="fixed")
    cm_adapt = ConformalMartingale(alpha=0.05, window=40, mode="adaptive")
    for x in prefix:
        cm_fixed.update(x)
        cm_adapt.update(x)
    for x in shifted:
        cm_fixed.update(x)
        cm_adapt.update(x)
    assert cm_fixed.alarmed  # persistent shift vs fixed bag: always small p
    # adaptive bag re-conforms: if it alarmed it did so during the burst
    if cm_adapt.alarmed:
        assert (cm_adapt.alarm_index or 0) < 40 + 120
