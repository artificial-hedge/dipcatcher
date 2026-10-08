"""Unit tests for quant_fund.models.vix_replication."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.models.vix_replication import (
    bench_vix_replication,
    corridor_strike,
    implied_forward,
    otm_strip,
    variance_strike,
)


def _bs_surface(sig: float = 0.22, s0: float = 100.0, r: float = 0.03, t: float = 0.08):
    fwd = s0 * np.exp(r * t)
    strikes = np.linspace(70.0, 140.0, 141)
    d1 = (np.log(fwd / strikes) + 0.5 * sig * sig * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)
    disc = np.exp(-r * t)
    calls = disc * (fwd * norm.cdf(d1) - strikes * norm.cdf(d2))
    puts = calls - disc * (fwd - strikes)
    return strikes, calls, puts, fwd, r, t


def test_implied_forward_recovers_parity() -> None:
    strikes, calls, puts, fwd, r, t = _bs_surface()
    f_hat, k0 = implied_forward(strikes, calls, puts, r, t)
    assert f_hat == pytest.approx(fwd, abs=0.2)
    assert k0 <= fwd


def test_variance_strike_matches_vol() -> None:
    strikes, calls, puts, fwd, r, t = _bs_surface(sig=0.22)
    f_hat, k0 = implied_forward(strikes, calls, puts, r, t)
    q = otm_strip(strikes, calls, puts, k0)
    var = variance_strike(strikes, q, f_hat, k0, r, t)
    assert np.sqrt(var) == pytest.approx(0.22, abs=0.01)


def test_otm_strip_uses_puts_below_calls_above() -> None:
    strikes, calls, puts, fwd, _, _ = _bs_surface()
    k0 = 99.9
    q = otm_strip(strikes, calls, puts, k0)
    np.testing.assert_allclose(q[strikes < k0], puts[strikes < k0])
    np.testing.assert_allclose(q[strikes > k0], calls[strikes > k0])


def test_corridor_below_full() -> None:
    strikes, calls, puts, _, r, t = _bs_surface()
    f_hat, k0 = implied_forward(strikes, calls, puts, r, t)
    q = otm_strip(strikes, calls, puts, k0)
    full = variance_strike(strikes, q, f_hat, k0, r, t)
    cor = corridor_strike(strikes, calls, puts, 85.0, 115.0, r, t)
    assert 0.0 < cor < full * 1.3


def test_rejects_bad_inputs() -> None:
    strikes, calls, puts, _, r, t = _bs_surface()
    with pytest.raises(ValueError):
        implied_forward(strikes[:2], calls[:2], puts[:2], r, t)
    with pytest.raises(ValueError):
        variance_strike(strikes, -np.abs(puts), 100.0, 99.0, r, t)
    with pytest.raises(ValueError):
        corridor_strike(strikes, calls, puts, 200.0, 300.0, r, t)


def test_bench_vix_replication_score() -> None:
    out = bench_vix_replication()
    assert out["synthetic_score"] == pytest.approx(1.0)
    assert out["synthetic_vix_sig_err"] < 0.02
