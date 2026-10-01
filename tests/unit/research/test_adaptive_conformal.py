"""Tests for research/adaptive_conformal.py — ACI under drift."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.adaptive_conformal import (
    ACI_SCHEMA,
    AdaptiveConformal,
    aci_run,
    adaptive_conformal_bench,
    static_conformal_width,
)


def test_aci_tracks_target_on_stationary_stream() -> None:
    """On iid half-normal residuals, long-run coverage approaches 1-alpha."""
    rng = np.random.default_rng(0)
    scores = np.abs(rng.normal(0.0, 1.0, 3000))
    out = aci_run(scores, alpha=0.1, step=0.05, calib_size=200)
    cov = float(np.mean(out["covered"][500:]))
    assert abs(cov - 0.9) < 0.03


def test_aci_recovers_after_scale_shift() -> None:
    """After a 3x variance step, ACI's alpha drops and coverage recovers."""
    rng = np.random.default_rng(1)
    n = 2000
    scores = np.abs(rng.normal(0.0, 1.0, n))
    scores[n // 2 :] *= 3.0
    out = aci_run(scores, alpha=0.1, step=0.08, calib_size=150)
    # coverage collapses at the shift, then recovers
    assert np.mean(out["covered"][n // 2 : n // 2 + 30]) < 0.85
    tail_cov = float(np.mean(out["covered"][-300:]))
    assert tail_cov > 0.85
    # ACI's fixed point is the alpha delivering realized target coverage
    # on the rolling window, not necessarily the nominal target — assert
    # only that alpha moved off its start in response to the shift
    assert out["alphas"][-1] != pytest.approx(0.1, abs=0.01)


def test_static_conformal_width_matches_quantile() -> None:
    scores = np.arange(1.0, 201.0)
    w = static_conformal_width(scores, 0.1, calib_size=100)
    assert np.isinf(w[0])
    # at t=150 the window is scores[50..149] -> q0.9 of 51..150
    expected = float(np.quantile(scores[50:150], 0.9))
    assert w[150] == pytest.approx(expected)


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        AdaptiveConformal(alpha=0.0)
    with pytest.raises(ValueError):
        AdaptiveConformal(alpha=0.1, step=-1.0)
    with pytest.raises(ValueError):
        AdaptiveConformal(alpha=0.1, calib_size=3)
    with pytest.raises(ValueError):
        AdaptiveConformal(alpha=0.1, alpha_bounds=(0.5, 0.1))
    a = AdaptiveConformal(alpha=0.1)
    with pytest.raises(ValueError):
        a.update(-1.0)
    with pytest.raises(ValueError):
        a.update(float("nan"))
    with pytest.raises(ValueError):
        aci_run(np.asarray([1.0, -1.0] * 20), alpha=0.1)
    with pytest.raises(ValueError):
        aci_run(np.ones(5), alpha=0.1)
    with pytest.raises(ValueError):
        adaptive_conformal_bench(n=50)


def test_bench_schema_and_claim() -> None:
    r = adaptive_conformal_bench(n=1000, alpha=0.1, seed=4)
    assert r["schema"] == ACI_SCHEMA
    assert r["kind"] == "adaptive_conformal"
    assert r["data_label"] == "SYNTHETIC"
    assert r["research_only"] is True
    # static miscalibrated post-shift; ACI recovers toward target
    assert r["static_mae_post"] > 0.03
    assert r["aci_mae_post"] < r["static_mae_post"]
    assert r["aci_recovery_time_events"] <= r["static_recovery_time_events"]
    assert len(r["payload_sha256"]) == 64


def test_bench_determinism() -> None:
    a = adaptive_conformal_bench(n=600, seed=11)
    b = adaptive_conformal_bench(n=600, seed=11)
    assert a["payload_sha256"] == b["payload_sha256"]
