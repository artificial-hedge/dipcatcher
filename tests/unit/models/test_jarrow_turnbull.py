"""Unit tests for quant_fund.models.jarrow_turnbull."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.jarrow_turnbull import (
    bench_jarrow_turnbull,
    calibrate_hazard,
    cds_par_spread,
    risky_bond_price,
    survival_prob,
)


def test_survival_piecewise_flat() -> None:
    lam = np.array([0.02, 0.05])
    times = np.array([1.0, 3.0])
    q = survival_prob(lam, times, np.array([0.5, 2.0]))
    assert q[0] == pytest.approx(np.exp(-0.01))
    assert q[1] == pytest.approx(np.exp(-0.02 - 0.05))


def test_risky_bond_limits() -> None:
    lam = np.array([0.02])
    times = np.array([5.0])
    # delta=0 -> pure survival leg Q(T) D(T).
    p0 = risky_bond_price(lam, times, 5.0, 0.0, 0.03)
    assert p0 == pytest.approx(np.exp(-(0.02 + 0.03) * 5.0), rel=1e-6)
    # delta=1 -> recovery paid at default: 0.02 (1-e^{-0.05*5})/0.05
    # + e^{-0.25} — exceeds the risk-free price.
    expect = 0.02 * (1 - np.exp(-0.05 * 5.0)) / 0.05 + np.exp(-0.25)
    p_rf = risky_bond_price(lam, times, 5.0, 1.0, 0.03)
    assert p_rf == pytest.approx(expect, rel=1e-9)


def test_calibration_recovers_hazard() -> None:
    lam_true = np.array([0.03, 0.04])
    times = np.array([2.0, 4.0])
    tp = np.array([risky_bond_price(lam_true, times, t, 0.4, 0.02) for t in times])
    lam_hat = calibrate_hazard(tp, times, 0.4, 0.02)
    assert lam_hat == pytest.approx(lam_true, abs=1e-6)


def test_cds_spread_positive_and_bounded() -> None:
    lam = np.array([0.02, 0.04, 0.06])
    times = np.array([1.0, 3.0, 5.0])
    s = cds_par_spread(lam, times, 0.4, 0.03)
    assert 0.001 < s < 0.1


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        survival_prob(np.array([0.02, -0.01]), np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        risky_bond_price(np.array([0.02]), np.array([5.0]), -1.0, 0.4, 0.03)
    with pytest.raises(ValueError):
        calibrate_hazard(np.array([2.0]), np.array([1.0]), 0.4, 0.03)


def test_bench_score() -> None:
    out = bench_jarrow_turnbull()
    assert out["score"] == 1.0
    assert out["synthetic_jt_calib_err"] < 1e-6
