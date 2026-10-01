"""Tests for microstructure/queue_depletion.py — gamma fill-time model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.microstructure.queue_depletion import (
    QUEUE_DEPLETION_SCHEMA,
    ProbeOutcome,
    collect_touch_probes,
    death_stats,
    empirical_fill_stats,
    fill_cdf,
    fill_quantile,
    fill_survival,
    max_cdf_gap,
    queue_depletion_bench,
)
from quant_fund.microstructure.zi_lob_simulator import santa_fe_config


def test_death_stats_closed_form() -> None:
    s = death_stats(4, 0.5)
    assert s["mean"] == pytest.approx(8.0)
    assert s["var"] == pytest.approx(16.0)


def test_fill_survival_gamma_kat() -> None:
    # q=1: S(t) = exp(-delta t); q=2: S = exp(-dt)(1 + dt)
    assert fill_survival(1, 2.0, 0.5) == pytest.approx(math.exp(-1.0))
    assert fill_survival(2, 1.0, 3.0) == pytest.approx(math.exp(-3.0) * (1.0 + 3.0))
    # t=0 -> 1, t->inf -> 0
    assert fill_survival(3, 1.0, 0.0) == 1.0
    assert fill_cdf(3, 1.0, 100.0) == pytest.approx(1.0, abs=1e-12)


def test_fill_quantile_inverts_cdf() -> None:
    for q in (1, 2, 5, 9):
        t50 = fill_quantile(q, 0.3, 0.5)
        assert fill_cdf(q, 0.3, t50) == pytest.approx(0.5, abs=1e-6)


def test_closed_form_fail_closed() -> None:
    with pytest.raises(ValueError):
        death_stats(0, 1.0)
    with pytest.raises(ValueError):
        death_stats(2, 0.0)
    with pytest.raises(ValueError):
        fill_survival(1, 1.0, -1.0)
    with pytest.raises(ValueError):
        fill_quantile(1, 1.0, 1.0)
    with pytest.raises(ValueError):
        fill_cdf(1.5, 1.0, 1.0)  # type: ignore[arg-type]


def test_empirical_stats() -> None:
    recs = [
        ProbeOutcome("buy", 1, 5.0, True),
        ProbeOutcome("buy", 1, 10.0, True),
        ProbeOutcome("buy", 1, 50.0, False),
        ProbeOutcome("sell", 3, 20.0, True),
    ]
    st = empirical_fill_stats(recs)
    assert st[1]["n"] == 3.0
    assert st[1]["mean_fill_time"] == pytest.approx(7.5)
    assert st[1]["fill_frac"] == pytest.approx(2 / 3)
    assert st[3]["n"] == 1.0
    with pytest.raises(ValueError):
        empirical_fill_stats([])


def test_collect_probes_smoke() -> None:
    cfg = santa_fe_config(seed=4)
    recs, rates = collect_touch_probes(cfg, horizon=300.0, probe_interval=30.0)
    assert recs, "no probes"
    assert all(r.queue_ahead >= 0 for r in recs)
    assert rates["mo_rate"] > 0.0
    assert any(r.event for r in recs)


def test_max_cdf_gap_bounds() -> None:
    rng = np.random.default_rng(0)
    # records under the model itself: q=2, delta=0.2
    recs = [
        ProbeOutcome("buy", 2, float(min(t, 60.0)), t <= 60.0) for t in rng.gamma(2, 1 / 0.2, 300)
    ]
    gap = max_cdf_gap(recs, 0.2, t_max=40.0)
    assert 0.0 <= gap < 0.4  # model-vs-model sampling noise bound


def test_bench_receipt_schema() -> None:
    r = queue_depletion_bench(horizon=800.0, probe_interval=40.0, seed=6)
    assert r["schema"] == QUEUE_DEPLETION_SCHEMA
    assert r["kind"] == "queue_depletion"
    assert r["data_label"] == "SYNTHETIC"
    assert r["research_only"] is True
    assert r["n_probes"] >= 20
    assert r["delta_hat"] > 0.0
    assert 0.0 <= r["max_cdf_gap"] <= 1.0
    assert len(r["payload_sha256"]) == 64


def test_bench_determinism_and_fail_closed() -> None:
    a = queue_depletion_bench(horizon=500.0, probe_interval=50.0, seed=9)
    b = queue_depletion_bench(horizon=500.0, probe_interval=50.0, seed=9)
    assert a["payload_sha256"] == b["payload_sha256"]
    with pytest.raises(ValueError):
        queue_depletion_bench(horizon=0.0)
    with pytest.raises(ValueError):
        queue_depletion_bench(t_max=0.0)
