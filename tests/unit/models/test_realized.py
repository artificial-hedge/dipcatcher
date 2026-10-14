"""Tests for realized volatility and jump detection."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.realized import (
    bipower_variation,
    bns_jump_test,
    lee_mykland_jumps,
    preaveraged_rv,
    realized_kernel,
    realized_quarticity,
    realized_variance,
    rv_confidence_band,
    semivariance,
    tripower_quarticity,
    tsrv,
)


def _bm_increments(n: int = 2000, sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    """Brownian increments with integrated variance ~ sigma^2."""
    rng = np.random.default_rng(seed)
    return rng.normal(scale=sigma / np.sqrt(n), size=n)


def _with_noise(r: np.ndarray, eta: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return r + rng.normal(scale=eta, size=r.size)


class TestBasicEstimators:
    def test_rv_recovers_iv(self):
        r = _bm_increments(n=1000, sigma=2.0)
        assert abs(realized_variance(r) - 4.0) / 4.0 < 0.15

    def test_bipower_under_jumps(self):
        r = _bm_increments(n=2000, sigma=1.0, seed=2)
        rv0 = realized_variance(r)
        bv0 = bipower_variation(r)
        assert abs(bv0 - rv0) / rv0 < 0.15
        # Inject 5 large jumps; BV stays ~IV, RV inflates.
        r2 = r.copy()
        rng = np.random.default_rng(3)
        idx = rng.choice(2000, 5, replace=False)
        r2[idx] += 0.4
        assert realized_variance(r2) > rv0 * 1.5
        assert abs(bipower_variation(r2) - bv0) / bv0 < 0.35

    def test_tripower_quarticity_positive(self):
        r = _bm_increments()
        assert tripower_quarticity(r) > 0
        assert realized_quarticity(r) > 0

    def test_semivariance(self):
        r = _bm_increments(n=1000, seed=4)
        out = semivariance(r)
        assert out["rs_pos"] > 0 and out["rs_neg"] > 0
        assert abs(out["rs_ratio"]) < 0.3
        # All-negative series: rs_pos = 0.
        out2 = semivariance(-np.abs(r))
        assert out2["rs_pos"] == 0.0
        assert out2["signed_jump"] < 0

    def test_failclosed(self):
        with pytest.raises(ValueError):
            realized_variance(np.ones(3))
        with pytest.raises(ValueError):
            bipower_variation(np.full(30, np.nan))


class TestJumpTests:
    def test_bns_detects_jumps(self):
        r = _bm_increments(n=3000, sigma=1.0, seed=5)
        out0 = bns_jump_test(r)
        assert out0["pvalue"] > 0.05
        r2 = r.copy()
        rng = np.random.default_rng(6)
        r2[rng.choice(3000, 8, replace=False)] += 0.25
        out1 = bns_jump_test(r2)
        assert out1["pvalue"] < 0.05
        assert out1["jump_share"] > 0.05

    def test_lee_mykland_flags_injected(self):
        r = _bm_increments(n=2000, sigma=1.0, seed=7)
        jump_at = np.array([500, 1000, 1500])
        r[jump_at] += 0.15  # ~7-sigma jump vs local bipower vol
        out = lee_mykland_jumps(r, alpha=0.999)
        found = set(out["jump_idx"].astype(int).tolist())
        assert len(found & set(jump_at.tolist())) >= 2
        # No jumps: few or zero detections.
        out0 = lee_mykland_jumps(_bm_increments(n=2000, seed=8), alpha=0.999)
        assert out0["is_jump"].sum() <= 3

    def test_failclosed(self):
        with pytest.raises(ValueError):
            bns_jump_test(np.zeros(50))
        with pytest.raises(ValueError):
            bns_jump_test(np.ones(50), alpha=0.1)


class TestNoiseRobust:
    def test_tsrv_reduces_noise_bias(self):
        # TSRV corrects *price-level* microstructure noise p_obs = p + u:
        # observed returns are MA(1), so add iid noise to the price path.
        n = 4000
        r = _bm_increments(n=n, sigma=1.0, seed=9)
        rng = np.random.default_rng(10)
        u = rng.normal(scale=0.01, size=n + 1)
        p_noisy = np.concatenate([[0.0], np.cumsum(r)]) + u
        rn = np.diff(p_noisy)
        rv = realized_variance(rn)
        t = tsrv(rn)
        true_iv = realized_variance(r)
        assert rv > true_iv * 1.5  # RV is heavily noise-biased (2 n sigma_u^2)
        assert abs(t - true_iv) < abs(rv - true_iv)

    def test_kernel_positive_and_stable(self):
        r = _with_noise(_bm_increments(n=1000, seed=11), 0.001, seed=12)
        k = realized_kernel(r, bandwidth=10)
        assert np.isfinite(k)
        rv = realized_variance(r)
        # Kernel should differ from naive RV in presence of noise.
        assert abs(k - rv) > 0.0

    def test_preaveraged_near_iv(self):
        n = 3000
        r = _bm_increments(n=n, sigma=1.5, seed=13)
        rng = np.random.default_rng(14)
        p_noisy = np.concatenate([[0.0], np.cumsum(r)]) + rng.normal(scale=0.002, size=n + 1)
        rn = np.diff(p_noisy)
        mrv = preaveraged_rv(rn, theta=0.4)
        true_iv = realized_variance(r)
        assert abs(mrv - true_iv) / true_iv < 0.5
        assert mrv < realized_variance(rn)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            tsrv(np.ones(10))
        with pytest.raises(ValueError):
            preaveraged_rv(np.ones(50), theta=-1.0)
        with pytest.raises(ValueError):
            realized_kernel(np.ones(100), bandwidth=0)


class TestBands:
    def test_rv_band_contains_iv(self):
        r = _bm_increments(n=2000, sigma=1.0, seed=15)
        out = rv_confidence_band(r, alpha=0.95)
        assert out["lower"] < out["rv"] < out["upper"]
        assert out["lower"] < 1.0 < out["upper"]

    def test_band_width_shrinks(self):
        r1 = _bm_increments(n=500, seed=16)
        r2 = _bm_increments(n=4000, seed=16)
        w1 = rv_confidence_band(r1)["upper"] - rv_confidence_band(r1)["lower"]
        w2 = rv_confidence_band(r2)["upper"] - rv_confidence_band(r2)["lower"]
        assert w2 < w1


def test_lee_mykland_window_matches_direct_bipower() -> None:
    """The vectorized window is the same bipower sum the scalar loop used."""
    r = _bm_increments(n=64, sigma=1.0, seed=21)
    out = lee_mykland_jumps(r, alpha=0.999)
    n = r.size
    k = max(2, int(np.floor(np.sqrt(n))))
    t = k + 3
    seg = r[t - k : t]
    direct = float(np.sum(np.abs(seg[1:]) * np.abs(seg[:-1])))
    sigma2 = direct * np.pi / (2.0 * (k - 1))
    stat = abs(float(r[t])) / np.sqrt(max(sigma2, 1e-16))
    assert out["stat"][t] == pytest.approx(stat, rel=0, abs=1e-12)


def test_kernel_lags_match_direct_dots() -> None:
    r = _bm_increments(n=80, seed=22)
    bandwidth = 6
    direct = 0.0
    rv = realized_variance(r)
    for h in range(1, bandwidth + 1):
        x = h / (bandwidth + 1.0)
        weight = 1.0 - 6.0 * x * x + 6.0 * x**3 if x <= 0.5 else 2.0 * (1.0 - x) ** 3
        direct += weight * float(r[h:] @ r[:-h])
    assert realized_kernel(r, bandwidth=bandwidth) == pytest.approx(
        rv + 2.0 * direct, rel=0, abs=1e-10
    )


def test_preaveraged_blocks_match_direct_dots() -> None:
    r = _bm_increments(n=40, seed=23)
    theta = 0.4
    n = r.size
    k_n = max(3, int(round(theta * np.sqrt(n))))
    weights = np.minimum(np.arange(1, k_n) / k_n, 1.0 - np.arange(1, k_n) / k_n)
    m = n - k_n + 1
    direct = np.array([float(weights @ r[t + 1 : t + k_n]) for t in range(m)])
    # Re-enter through the public estimator's building block by comparing the
    # modulated sum, which is what the estimator squares.
    from quant_fund.models.realized import preaveraged_rv as impl

    got = impl(r, theta=theta)
    psi2 = 1.0 / 12.0
    mrv = (n / m) * float(direct @ direct) / (psi2 * k_n)
    expected = mrv - realized_variance(r) / (2.0 * theta**2 * psi2 * n)
    assert got == pytest.approx(expected, rel=0, abs=1e-10)
