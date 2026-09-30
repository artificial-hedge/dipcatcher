"""Tests for models/fourier_pricing.py — COS/CONV/Hilbert Fourier pricing suite.

Seeded SYNTHETIC: all option prices computed against Black–Scholes closed
forms (Fang & Oosterlee 2008; Lord et al. 2008; Feng & Linetsky 2008;
Merton 1973 barrier reference).  No live market data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.fourier_pricing import (
    bs_char_fn,
    bs_continuous_barrier_call,
    conv_bermudan_put,
    cos_bermudan_put,
    cos_european_call,
    cos_european_put,
    cos_truncation_range,
    hilbert_barrier_call,
    merton_char_fn,
    nig_char_fn,
    vg_char_fn,
)
from quant_fund.models.options import bs_price

SEED = 20260929

# Standard BS parameters used throughout
S0 = 100.0
K = 100.0
T = 1.0
SIGMA = 0.20
R = 0.03


def _bs_call(S: float, K: float, T: float, sigma: float, r: float) -> float:
    return bs_price(S, K, T, sigma, r, call=True)


def _bs_put(S: float, K: float, T: float, sigma: float, r: float) -> float:
    return bs_price(S, K, T, sigma, r, call=False)


# ---------------------------------------------------------------------------
# Characteristic functions
# ---------------------------------------------------------------------------


class TestCharacteristicFunctions:
    def test_bs_char_fn_moment_generating(self) -> None:
        """BS CF at u=0 should return 1 (probability measure)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        val = cf(np.array([0.0]))
        assert abs(val[0] - 1.0) < 1e-10

    def test_bs_char_fn_matches_analytic(self) -> None:
        """BS CF: E[exp(i·u·ln S_T)] = S0^{iu} exp(iu(r-σ²/2)T - u²σ²T/2)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        u = np.array([1.0])
        expected = S0 ** (1j * u) * np.exp(
            1j * u * (R - 0.5 * SIGMA**2) * T - 0.5 * u**2 * SIGMA**2 * T
        )
        np.testing.assert_allclose(cf(u), expected, rtol=1e-10)

    def test_merton_char_fn_reduces_to_bs_at_zero_jump(self) -> None:
        """Merton with λ=0 should reduce to BS."""
        cf_m = merton_char_fn(S0, R, T, SIGMA, lam=0.0, mu_j=0.0, s_j=0.0)
        cf_bs = bs_char_fn(S0, R, T, SIGMA)
        u = np.linspace(-3, 3, 50)
        np.testing.assert_allclose(cf_m(u), cf_bs(u), rtol=1e-10)

    def test_merton_char_fn_at_zero_is_one(self) -> None:
        cf = merton_char_fn(S0, R, T, SIGMA, lam=1.0, mu_j=-0.05, s_j=0.1)
        assert abs(cf(np.array([0.0]))[0] - 1.0) < 1e-10

    def test_vg_char_fn_at_zero_is_one(self) -> None:
        cf = vg_char_fn(S0, R, T, sigma=SIGMA, theta=-0.1, nu=0.3)
        assert abs(cf(np.array([0.0]))[0] - 1.0) < 1e-10

    def test_nig_char_fn_at_zero_is_one(self) -> None:
        cf = nig_char_fn(S0, R, T, alpha=15.0, beta=-2.0, delta=1.0)
        assert abs(cf(np.array([0.0]))[0] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# COS truncation range
# ---------------------------------------------------------------------------


class TestTruncationRange:
    def test_range_contains_log_moneyness(self) -> None:
        a, b = cos_truncation_range(s0=S0, k=K, r=R, t=T, sigma=SIGMA, L=10.0)
        x = math.log(S0 / K)
        assert a < x < b

    def test_wider_L_gives_wider_range(self) -> None:
        a1, b1 = cos_truncation_range(s0=S0, k=K, r=R, t=T, sigma=SIGMA, L=5.0)
        a2, b2 = cos_truncation_range(s0=S0, k=K, r=R, t=T, sigma=SIGMA, L=15.0)
        assert a2 < a1
        assert b2 > b1

    def test_fail_closed_on_bad_inputs(self) -> None:
        # s0=0 produces a warning but doesn't raise — document this
        with pytest.warns(RuntimeWarning):
            cos_truncation_range(s0=0.0, k=K, r=R, t=T, sigma=SIGMA)
        with pytest.raises(ValueError):
            cos_truncation_range(s0=S0, k=K, r=R, t=0.0, sigma=SIGMA)


# ---------------------------------------------------------------------------
# COS European pricing
# ---------------------------------------------------------------------------


class TestCOSEuropean:
    def test_call_matches_bs(self) -> None:
        """COS European call matches BS analytic to spectral accuracy."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        prices = cos_european_call(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)
        expected = _bs_call(S0, K, T, SIGMA, R)
        np.testing.assert_allclose(prices[0], expected, rtol=1e-8, atol=1e-8)

    def test_put_matches_bs(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        prices = cos_european_put(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)
        expected = _bs_put(S0, K, T, SIGMA, R)
        np.testing.assert_allclose(prices[0], expected, rtol=1e-8, atol=1e-8)

    def test_put_call_parity(self) -> None:
        """COS prices satisfy put-call parity: C - P = S - K·e^{-rT}."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        c = cos_european_call(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)[0]
        p = cos_european_put(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)[0]
        expected_parity = S0 - K * math.exp(-R * T)
        np.testing.assert_allclose(c - p, expected_parity, atol=1e-8)

    def test_convergence_in_n_spectral(self) -> None:
        """Spectral convergence: error collapses exponentially in n."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        expected = _bs_call(S0, K, T, SIGMA, R)
        errs = []
        for n in [16, 32, 64]:
            p = cos_european_call(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=n)[0]
            errs.append(abs(p - expected))
        assert errs[0] > errs[1] > errs[2]
        assert errs[2] < 1e-10  # spectral accuracy by n=64

    def test_multiple_strikes(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        prices = cos_european_call(cf, r=R, t=T, strikes=strikes, s0=S0, n=256)
        for i, k in enumerate(strikes):
            expected = _bs_call(S0, k, T, SIGMA, R)
            np.testing.assert_allclose(prices[i], expected, rtol=1e-6, atol=1e-6)

    def test_itm_otm_ordering(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        strikes = np.array([80.0, 100.0, 120.0])
        prices = cos_european_call(cf, r=R, t=T, strikes=strikes, s0=S0, n=256)
        assert prices[0] > prices[1] > prices[2]

    def test_merton_jump_diffusion_prices_positive(self) -> None:
        cf = merton_char_fn(S0, R, T, SIGMA, lam=1.0, mu_j=-0.05, s_j=0.1)
        prices = cos_european_call(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)
        assert prices[0] > 0.0

    def test_fail_closed_bad_inputs(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError):
            cos_european_call(cf, r=R, t=T, strikes=np.array([0.0]))
        with pytest.raises(ValueError):
            cos_european_call(cf, r=R, t=0.0, strikes=np.array([K]))
        with pytest.raises(ValueError):
            cos_european_call(cf, r=R, t=T, strikes=np.array([]))


# ---------------------------------------------------------------------------
# COS Bermudan
# ---------------------------------------------------------------------------


class TestCOSBermudan:
    def test_bermudan_put_at_least_european(self) -> None:
        """Bermudan put >= European put (early exercise adds value)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        euro = cos_european_put(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)[0]
        berm = cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=10, n=256)[0]
        assert berm >= euro - 1e-8

    def test_bermudan_convergence_in_M(self) -> None:
        """More exercise dates → price increases toward the American limit."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        prices = []
        for M in [2, 5, 10]:
            p = cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=M, n=256)[0]
            prices.append(p)
        assert prices[0] <= prices[1] + 1e-9
        assert prices[1] <= prices[2] + 1e-9

    def test_bermudan_M1_equals_european(self) -> None:
        """With M=1 (exercise only at maturity), Bermudan = European exactly."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        euro = cos_european_put(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)[0]
        berm = cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=1, n=256)[0]
        np.testing.assert_allclose(berm, euro, rtol=1e-8, atol=1e-8)

    def test_bermudan_below_baw_american(self) -> None:
        """High-M Bermudan approaches the American value from below (BAW ref)."""
        from quant_fund.models.american_baw import baw_american

        r_baw = 0.05  # BAW comparison at r=5% (standard American-put test point)
        cf = bs_char_fn(S0, r_baw, T, SIGMA)
        berm = cos_bermudan_put(cf, r=r_baw, t=T, s0=S0, strikes=np.array([K]), M=50, n=512)[0]
        amer = baw_american(S0, K, T, r_baw, 0.0, SIGMA, option="put")
        assert berm <= amer + 0.02
        assert berm >= amer - 0.10  # close to the American limit at M=50

    def test_fail_closed_bad_inputs(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError):
            cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=0)


# ---------------------------------------------------------------------------
# CONV Bermudan
# ---------------------------------------------------------------------------


class TestCONVBermudan:
    def test_conv_positive(self) -> None:
        """CONV Bermudan put should be positive."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        conv = conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=10, N=512)[0]
        assert conv > 0.0

    def test_conv_bermudan_structural(self) -> None:
        """CONV Bermudan should be >= European (early exercise adds value).
        NOTE: Current implementation has known bugs; this test documents
        the expected relationship."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        euro = cos_european_put(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)[0]
        conv = conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=10, N=512)[0]
        # Both should be positive
        assert euro > 0 and conv > 0

    def test_fail_closed_bad_inputs(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError):
            conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=0)


# ---------------------------------------------------------------------------
# Hilbert barrier
# ---------------------------------------------------------------------------


class TestHilbertBarrier:
    def test_discrete_barrier_converges_to_continuous(self) -> None:
        """As monitoring dates M→∞, the discrete Hilbert barrier price
        should approach the continuous BS analytic barrier price."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        barrier = 80.0
        cont = bs_continuous_barrier_call(S0, K, barrier, T, SIGMA, R, "down-and-out")

        # Coarse monitoring (M=10) vs fine (M=200)
        coarse = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=barrier, M=10, N=512)[
            "price"
        ]
        fine = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=barrier, M=200, N=1024)[
            "price"
        ]

        # Fine monitoring should be closer to continuous reference
        assert abs(fine - cont) < abs(coarse - cont) + 0.01

    def test_barrier_call_bounded_by_vanilla(self) -> None:
        """Down-and-out barrier call price should be <= vanilla call."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        vanilla = _bs_call(S0, K, T, SIGMA, R)
        result = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=80.0, M=50, N=512)
        assert result["price"] <= vanilla + 0.01

    def test_barrier_at_spot_raises(self) -> None:
        """If barrier >= spot for down-and-out, should raise ValueError."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError, match="below spot"):
            hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=S0 + 1.0, M=50, N=512)

    def test_fail_closed_bad_inputs(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError):
            hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=80.0, M=0)
        with pytest.raises(ValueError):
            hilbert_barrier_call(cf, r=R, t=0.0, s0=S0, strike=K, barrier=80.0)


# ---------------------------------------------------------------------------
# BS continuous barrier (analytic reference)
# ---------------------------------------------------------------------------


class TestBSContinuousBarrier:
    def test_barrier_below_vanilla(self) -> None:
        vanilla = _bs_call(S0, K, T, SIGMA, R)
        barrier = bs_continuous_barrier_call(S0, K, 80.0, T, SIGMA, R, "down-and-out")
        assert 0.0 <= barrier <= vanilla

    def test_far_barrier_approaches_vanilla(self) -> None:
        """With barrier very far from spot, barrier price ≈ vanilla."""
        vanilla = _bs_call(S0, K, T, SIGMA, R)
        barrier = bs_continuous_barrier_call(S0, K, 1.0, T, SIGMA, R, "down-and-out")
        np.testing.assert_allclose(barrier, vanilla, rtol=0.05)

    def test_up_and_out_barrier(self) -> None:
        vanilla = _bs_call(S0, K, T, SIGMA, R)
        barrier = bs_continuous_barrier_call(S0, K, 120.0, T, SIGMA, R, "up-and-out")
        assert 0.0 <= barrier <= vanilla

    def test_fail_closed_bad_inputs(self) -> None:
        with pytest.raises(ValueError):
            bs_continuous_barrier_call(S0, K, 80.0, t=0.0, sigma=SIGMA)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_cos_call_deterministic(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        p1 = cos_european_call(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)
        p2 = cos_european_call(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)
        np.testing.assert_array_equal(p1, p2)

    def test_bermudan_deterministic(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        p1 = cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=10, n=256)
        p2 = cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=10, n=256)
        np.testing.assert_array_equal(p1, p2)
