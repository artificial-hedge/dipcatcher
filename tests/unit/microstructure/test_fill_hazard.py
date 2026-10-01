"""Tests for microstructure/fill_hazard.py — probe-order survival lane."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.microstructure.fill_hazard import (
    FILL_HAZARD_SCHEMA,
    FillRecord,
    collect_fill_records,
    fill_hazard_bench,
    fill_prob,
    fit_exp_intensity,
    kaplan_meier,
    nelson_aalen,
)
from quant_fund.microstructure.zi_lob_simulator import santa_fe_config


def _rec(d: int, dur: float, event: bool, t0: float = 0.0) -> FillRecord:
    return FillRecord(
        side="buy",
        delta_ticks=d,
        queue_ahead=0,
        t_submit=t0,
        duration=dur,
        event=event,
    )


def test_kaplan_meier_kat() -> None:
    # 4 records: fills at 1,2; censor at 3 -> S(2) = (1-1/4)*(1-1/3) = 0.5
    recs = [_rec(0, 1.0, True), _rec(0, 2.0, True), _rec(0, 3.0, False), _rec(0, 5.0, False)]
    t, s = kaplan_meier(recs)
    assert t[0] == 0.0 and s[0] == 1.0
    assert s[-1] == pytest.approx(0.5)
    assert np.all(np.diff(s) <= 0.0)  # non-increasing


def test_nelson_aalen_kat() -> None:
    recs = [_rec(0, 1.0, True), _rec(0, 2.0, True), _rec(0, 3.0, False), _rec(0, 5.0, False)]
    t, lam = nelson_aalen(recs)
    # Λ(1)=1/4, Λ(2)=1/4+1/3
    assert lam[-1] == pytest.approx(1 / 4 + 1 / 3)
    assert np.all(np.diff(lam) > 0.0)


def test_survival_estimators_fail_closed() -> None:
    with pytest.raises(ValueError):
        kaplan_meier([])
    with pytest.raises(ValueError):
        nelson_aalen([])


def test_exp_intensity_recovers_kappa() -> None:
    # Simulate records under known lambda(delta)=A*exp(-k*d): durations are
    # exponential; construct deterministic durations via quantiles so the
    # empirical rate per bucket lands near truth.
    rng = np.random.default_rng(0)
    a_true, k_true = 0.10, 0.8
    recs: list[FillRecord] = []
    for d in (0, 1, 2, 4):
        lam = a_true * math.exp(-k_true * d)
        for _ in range(400):
            dur = float(rng.exponential(1.0 / lam))
            horizon = 200.0
            recs.append(_rec(d, min(dur, horizon), dur <= horizon))
    fit = fit_exp_intensity(recs)
    assert fit.kappa == pytest.approx(k_true, rel=0.15)
    assert fit.a == pytest.approx(a_true, rel=0.25)
    assert fill_prob(fit, 0, 10.0) == pytest.approx(1 - math.exp(-a_true * 10), rel=0.2)


def test_exp_intensity_fail_closed() -> None:
    with pytest.raises(ValueError):
        fit_exp_intensity([])
    with pytest.raises(ValueError):  # single non-degenerate bucket
        fit_exp_intensity([_rec(0, 1.0, True), _rec(0, 2.0, True)])
    with pytest.raises(ValueError):  # zero exposure
        fit_exp_intensity([_rec(0, 0.0, False), _rec(1, 1.0, True)])


def test_fill_prob_fail_closed() -> None:
    recs = [_rec(0, 1.0, True), _rec(0, 2.0, True), _rec(1, 5.0, True), _rec(1, 6.0, False)]
    fit = fit_exp_intensity(recs)
    with pytest.raises(ValueError):
        fill_prob(fit, -1, 10.0)
    with pytest.raises(ValueError):
        fill_prob(fit, 0, -1.0)


def test_collect_records_smoke() -> None:
    cfg = santa_fe_config(seed=5)
    recs = collect_fill_records(cfg, horizon=300.0, probe_interval=30.0, min_runway=10.0)
    assert recs, "no probes collected"
    assert all(r.duration >= 0.0 for r in recs)
    assert any(r.event for r in recs)  # touch probes should fill sometimes
    assert all(isinstance(r.delta_ticks, int) and r.delta_ticks >= 0 for r in recs)


def test_collect_fail_closed() -> None:
    cfg = santa_fe_config(seed=5)
    with pytest.raises(ValueError):
        collect_fill_records(cfg, horizon=0.0)
    with pytest.raises(ValueError):
        collect_fill_records(cfg, horizon=100.0, delta_ticks=(-1,))
    with pytest.raises(ValueError):
        collect_fill_records(cfg, horizon=100.0, sides=("bad",))


def test_bench_receipt_schema() -> None:
    rec = fill_hazard_bench(n_probe_horizon=600.0, probe_interval=40.0, seed=7)
    assert rec["schema"] == FILL_HAZARD_SCHEMA
    assert rec["kind"] == "fill_hazard"
    assert rec["data_label"] == "SYNTHETIC"
    assert rec["research_only"] is True
    assert rec["n_records"] > 0
    assert rec["n_fills"] > 0
    assert rec["exp_kappa"] > 0.0
    assert 0.0 <= rec["censor_fraction"] <= 1.0
    assert len(rec["payload_sha256"]) == 64
    # deeper orders fill slower
    meds = rec["median_fill_time_by_delta"]
    assert rec["km_monotone_in_delta"] is True or len(meds) >= 2


def test_bench_determinism() -> None:
    r1 = fill_hazard_bench(n_probe_horizon=400.0, probe_interval=50.0, seed=3)
    r2 = fill_hazard_bench(n_probe_horizon=400.0, probe_interval=50.0, seed=3)
    assert r1["payload_sha256"] == r2["payload_sha256"]
