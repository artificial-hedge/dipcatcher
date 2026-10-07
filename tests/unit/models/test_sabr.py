"""Adversarial probes for sabr."""

import numpy as np
import pytest

from quant_fund.models import sabr


def test_shift_rejects_nonfinite():
    with pytest.raises(ValueError, match="shift"):
        sabr.sabr_implied_vol(1.0, 1.0, 1.0, 0.3, 0.5, -0.3, 0.4, shift=np.nan)
    with pytest.raises(ValueError, match="shift"):
        sabr.sabr_implied_vol(1.0, 1.0, 1.0, 0.3, 0.5, -0.3, 0.4, shift=np.inf)


def test_atm_continuity():
    f, t, a, b, r, n = 100.0, 1.0, 0.2, 0.5, -0.3, 0.4
    atm = sabr.sabr_atm_vol(f, t, a, b, r, n)
    for dk in (-1e-6, 1e-6):
        near = float(sabr.sabr_implied_vol(f, f + dk, t, a, b, r, n))
        assert near == pytest.approx(atm, rel=1e-4)


def test_alpha_from_atm_inverts_atm_vol():
    f, t, a, b, r, n = 80.0, 0.75, 0.25, 0.7, -0.4, 0.6
    v = sabr.sabr_atm_vol(f, t, a, b, r, n)
    a_hat = sabr.sabr_alpha_from_atm(f, t, v, b, r, n)
    assert a_hat == pytest.approx(a, rel=1e-6)


def test_fit_recovers_params_on_self_smile():
    f, t, a, b, r, n = 100.0, 1.0, 0.3, 0.5, -0.25, 0.5
    k = np.linspace(70, 140, 12)
    v = sabr.sabr_implied_vol(f, k, t, a, b, r, n)
    out = sabr.sabr_fit(f, t, k, v, beta=0.5)
    assert out["rmse"] < 1e-6
    assert out["alpha"] == pytest.approx(a, rel=0.05)
    assert out["rho"] == pytest.approx(r, abs=0.1)
    assert out["nu"] == pytest.approx(n, rel=0.15)


def test_param_fail_closed():
    with pytest.raises(ValueError):
        sabr.sabr_implied_vol(1.0, 1.0, 1.0, -0.1, 0.5, 0.0, 0.4)
    with pytest.raises(ValueError):
        sabr.sabr_implied_vol(1.0, 1.0, 1.0, 0.3, 1.5, 0.0, 0.4)
    with pytest.raises(ValueError):
        sabr.sabr_implied_vol(1.0, 1.0, 1.0, 0.3, 0.5, 1.0, 0.4)
    with pytest.raises(ValueError):
        sabr.sabr_implied_vol(1.0, 1.0, 1.0, 0.3, 0.5, 0.0, 0.0)
    with pytest.raises(ValueError):
        sabr.sabr_implied_vol(1.0, 1.0, 0.0, 0.3, 0.5, 0.0, 0.4)


def test_shifted_sabr_negative_strikes():
    out = sabr.sabr_implied_vol(
        -0.01, np.array([-0.02, 0.0, 0.02]), 1.0, 0.01, 0.5, -0.2, 0.5, shift=0.05
    )
    assert np.all(np.isfinite(out))
    assert np.all(out > 0.0)


def test_smile_shape_monotone_skew():
    f, t, a, b, n = 1.0, 1.0, 0.3, 0.5, 0.5
    k = np.linspace(0.7, 1.4, 15)
    neg = sabr.sabr_implied_vol(f, k, t, a, b, -0.6, n)
    pos = sabr.sabr_implied_vol(f, k, t, a, b, 0.6, n)
    # negative rho => downward sloping smile (low strikes bid)
    assert float(neg[0]) > float(neg[-1])
    assert float(pos[-1]) > float(pos[0])
