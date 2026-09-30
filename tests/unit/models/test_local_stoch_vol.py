"""Tests for models/local_stoch_vol.py — Dupire local vol + SLV mixing MC.

SYNTHETIC: every surface and path here is model-generated (closed-form SABR /
Heston or seeded Monte Carlo), never market data. Assertions are correctness
checks of the Dupire round trip, the van der Stoep et al. (2014) mixing-MC
leverage scheme, and degenerate limits — not market evidence.

References exercised: Dupire (1994) Risk 7(1); Gatheral (2006) ch. 11;
van der Stoep, Grzelak & Oosterlee (2014) arXiv:1211.2993; Heston (1993).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.models.local_stoch_vol import (
    LocalVolSurface,
    dupire_local_vol,
    heston_mc,
    local_vol_mc,
    make_leverage_interpolator,
    mc_implied_vols,
    slv_leverage_function,
    slv_price_calls,
)
from quant_fund.models.options import bs_price, implied_vol
from quant_fund.models.sabr import sabr_implied_vol
from quant_fund.quant_models.heston import heston_call, heston_implied_vol

SPOT, R, Q = 100.0, 0.03, 0.0
SABR = dict(alpha=2.0, beta=0.5, rho=-0.4, nu=0.35)
HESTON = dict(kappa=3.0, theta=0.04, xi=0.30, rho=-0.5, v0=0.04)
STRIKES = np.linspace(70.0, 130.0, 25)
MATURITIES = np.array([0.25, 0.5, 0.75, 1.0, 1.25])
CALIB_STRIKES = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])


def _black_price(f: np.ndarray, k: np.ndarray, t: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Discounted Black-76 call price (broadcasts)."""
    d1 = (np.log(f / k) + 0.5 * sig**2 * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)
    return np.exp(-R * t) * (f * norm.cdf(d1) - k * norm.cdf(d2))


def _forwards(mats: np.ndarray) -> np.ndarray:
    return SPOT * np.exp((R - Q) * mats)


def _sabr_iv_grid() -> np.ndarray:
    f = _forwards(MATURITIES)
    return np.array(
        [sabr_implied_vol(fj, STRIKES, tj, **SABR) for fj, tj in zip(f, MATURITIES, strict=True)]
    )


def _sabr_surface() -> np.ndarray:
    """SYNTHETIC call surface: Hagan (2002) SABR vols priced through Black-76."""
    f = _forwards(MATURITIES)
    return _black_price(f[:, None], STRIKES[None, :], MATURITIES[:, None], _sabr_iv_grid())


def _moneyness_band(lo: float = 0.85, hi: float = 1.15) -> np.ndarray:
    f = _forwards(MATURITIES)
    return (STRIKES[None, :] >= lo * f[:, None]) & (STRIKES[None, :] <= hi * f[:, None])


def _flat_surface(sig: float) -> LocalVolSurface:
    return LocalVolSurface(
        np.array([70.0, 85.0, 100.0, 115.0, 130.0]),
        np.array([0.25, 0.5, 1.0]),
        np.full((3, 5), sig),
    )


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


@pytest.mark.synthetic
def test_dupire_recovers_constant_vol_black_surface() -> None:
    """Exact BS surface with flat sigma must give sigma_L == sigma (Dupire 1994)."""
    sig = 0.25
    prices = np.array([[bs_price(SPOT, k, T, sig, R) for k in STRIKES] for T in MATURITIES])
    surf = dupire_local_vol(STRIKES, MATURITIES, prices, spot=SPOT, r=R, q=Q)
    f = _forwards(MATURITIES)
    band = (STRIKES[None, :] >= 0.85 * f[:, None]) & (STRIKES[None, :] <= 1.15 * f[:, None])
    err = np.abs(surf.local_vol - sig)
    assert err[band].max() < 0.01
    assert surf.diagnostics["degenerate_frac"] == 0.0
    assert surf.diagnostics["frac_clamped_low"] == 0.0
    assert surf.diagnostics["frac_clamped_high"] == 0.0


@pytest.mark.synthetic
def test_dupire_roundtrip_reproduces_sabr_implied_vols() -> None:
    """Fundamental Dupire consistency: LV-MC implied vols reproduce the input surface."""
    clean = _sabr_surface()
    surf = dupire_local_vol(STRIKES, MATURITIES, clean, spot=SPOT, r=R, q=Q)
    mc = local_vol_mc(
        surf, spot=SPOT, r=R, q=Q, strikes=STRIKES, seed=11, n_paths=20_000, max_dt=1.0 / 24.0
    )
    iv_mc = mc_implied_vols(
        np.asarray(mc["prices"]), spot=SPOT, r=R, q=Q, strikes=STRIKES, maturities=MATURITIES
    )
    err = np.abs(iv_mc - _sabr_iv_grid())
    band = _moneyness_band()
    assert err[band].max() < 0.02  # 2 vol points across the central band
    assert err[band].mean() < 0.01
    assert mc["synthetic"] is True
    assert "SYNTHETIC" in str(mc["label"])


@pytest.mark.synthetic
def test_dupire_roundtrip_heston_cf_surface() -> None:
    """Same round trip on an arbitrage-free Heston CF surface (repo pricer)."""
    ks = np.linspace(75.0, 125.0, 15)
    mats = np.array([0.25, 0.5, 1.0])
    prices = np.array([[heston_call(k, T, S0=SPOT, r=R, q=Q, **HESTON) for k in ks] for T in mats])
    surf = dupire_local_vol(ks, mats, prices, spot=SPOT, r=R, q=Q)
    mc = local_vol_mc(
        surf, spot=SPOT, r=R, q=Q, strikes=ks, seed=12, n_paths=20_000, max_dt=1.0 / 24.0
    )
    iv_mc = mc_implied_vols(
        np.asarray(mc["prices"]), spot=SPOT, r=R, q=Q, strikes=ks, maturities=mats
    )
    # Invert the CF surface with the same inverter: the comparison isolates the
    # Dupire FD + LV-MC pipeline, not the inverter.
    iv_ref = mc_implied_vols(prices, spot=SPOT, r=R, q=Q, strikes=ks, maturities=mats)
    f = _forwards(mats)
    band = (ks[None, :] >= 0.85 * f[:, None]) & (ks[None, :] <= 1.15 * f[:, None])
    assert np.abs(iv_mc - iv_ref)[band].max() < 0.02


@pytest.mark.synthetic
def test_dupire_smoothing_stabilizes_noisy_surface() -> None:
    """Raw FD on a noisy surface explodes or fails closed; Tikhonov stays stable."""
    clean = _sabr_surface()
    ref = dupire_local_vol(STRIKES, MATURITIES, clean, spot=SPOT, r=R, q=Q)
    rng = np.random.default_rng(3)
    noisy = clean + rng.normal(0.0, 0.04, size=clean.shape)

    # Fail-closed at the default degeneracy/arbitrage tolerances.
    with pytest.raises(ValueError, match="degenerate|no-arbitrage"):
        dupire_local_vol(STRIKES, MATURITIES, noisy, spot=SPOT, r=R, q=Q)

    raw = dupire_local_vol(
        STRIKES,
        MATURITIES,
        noisy,
        spot=SPOT,
        r=R,
        q=Q,
        lambda_k=0.0,
        arb_tol=0.2,
        max_degenerate_frac=1.0,
    )
    smoothed = dupire_local_vol(
        STRIKES,
        MATURITIES,
        noisy,
        spot=SPOT,
        r=R,
        q=Q,
        arb_tol=0.2,
        max_degenerate_frac=0.35,
    )
    rmse_raw = _rmse(raw.local_vol, ref.local_vol)
    rmse_smooth = _rmse(smoothed.local_vol, ref.local_vol)
    assert np.isfinite(smoothed.local_vol).all()
    assert smoothed.local_vol.min() >= np.sqrt(2.5e-5)  # variance floor respected
    assert rmse_smooth < rmse_raw  # regularization strictly improves stability
    assert smoothed.diagnostics["tikhonov_rms_delta"] > 0.0


@pytest.mark.synthetic
def test_dupire_fail_closed_on_bad_input() -> None:
    clean = _sabr_surface()
    with pytest.raises(ValueError, match="strictly increasing"):
        dupire_local_vol(STRIKES[::-1], MATURITIES, clean[:, ::-1], spot=SPOT, r=R)
    with pytest.raises(ValueError, match="shape"):
        dupire_local_vol(STRIKES, MATURITIES, clean.T, spot=SPOT, r=R)
    dented = clean.copy()
    dented[:, 12] -= 0.5  # break butterfly convexity near ATM
    with pytest.raises(ValueError, match="no-arbitrage"):
        dupire_local_vol(STRIKES, MATURITIES, dented, spot=SPOT, r=R)
    intrinsic = np.maximum(
        SPOT * np.exp(-Q * MATURITIES)[:, None]
        - STRIKES[None, :] * np.exp(-R * MATURITIES)[:, None],
        0.0,
    )
    with pytest.raises(ValueError, match="degenerate"):
        dupire_local_vol(STRIKES, MATURITIES, intrinsic, spot=SPOT, r=R)
    with pytest.raises(ValueError):
        dupire_local_vol(STRIKES, MATURITIES, clean, spot=SPOT, r=R, lambda_k=-1.0)
    with pytest.raises(ValueError):
        dupire_local_vol(STRIKES[:4], MATURITIES, clean[:, :4], spot=SPOT, r=R)
    with pytest.raises(ValueError):
        dupire_local_vol(STRIKES, MATURITIES, clean, spot=SPOT, r=R, var_cap=1e-6)


def test_local_vol_surface_validation_and_clamping() -> None:
    with pytest.raises(ValueError):
        LocalVolSurface(np.array([100.0, 90.0]), np.array([0.5, 1.0]), np.ones((2, 2)))
    with pytest.raises(ValueError):
        LocalVolSurface(STRIKES[:5], MATURITIES[:3], np.zeros((3, 5)))
    with pytest.raises(ValueError):
        LocalVolSurface(STRIKES[:5], MATURITIES[:3], np.ones((5, 3)))
    surf = _flat_surface(0.25)
    with pytest.raises(ValueError):
        surf.vol(np.array([-1.0]), 0.5)
    with pytest.raises(ValueError):
        surf.vol(np.array([100.0]), -0.5)
    # constant extrapolation (clamped to the grid box) in both dimensions
    assert surf.vol(np.array([1e6]), 0.5) == pytest.approx(0.25)
    assert surf.vol(np.array([100.0]), 0.0) == pytest.approx(0.25)
    assert surf.vol(np.array([100.0]), 10.0) == pytest.approx(0.25)
    assert np.abs(surf.dvol_dlns(np.array([100.0]), 0.5)) < 1e-12


def test_mc_implied_vols_roundtrip() -> None:
    """BS prices -> implied vols recovers the generating vols (inverter check)."""
    sigs = np.array([0.15, 0.25, 0.35])
    mats = np.array([0.25, 0.5, 1.0])
    prices = np.array(
        [
            [bs_price(SPOT, k, T, s, R) for k in CALIB_STRIKES]
            for T, s in zip(mats, sigs, strict=True)
        ]
    )
    iv = mc_implied_vols(prices, spot=SPOT, r=R, q=0.0, strikes=CALIB_STRIKES, maturities=mats)
    assert np.allclose(iv, sigs[:, None], atol=1e-6)
    with pytest.raises(ValueError, match="shape"):
        mc_implied_vols(prices.T, spot=SPOT, r=R, strikes=CALIB_STRIKES, maturities=mats)
    with pytest.raises(ValueError):
        bad = prices.copy()
        bad[0, 0] = -1.0
        mc_implied_vols(bad, spot=SPOT, r=R, strikes=CALIB_STRIKES, maturities=mats)


def test_local_vol_mc_determinism_and_fail_closed() -> None:
    surf = _flat_surface(0.25)
    kw = dict(spot=SPOT, r=R, q=Q, strikes=CALIB_STRIKES, n_paths=2_000, max_dt=0.25)
    a = local_vol_mc(surf, seed=7, **kw)
    b = local_vol_mc(surf, seed=7, **kw)
    c = local_vol_mc(surf, seed=8, **kw)
    assert np.array_equal(np.asarray(a["prices"]), np.asarray(b["prices"]))
    assert not np.array_equal(np.asarray(a["prices"]), np.asarray(c["prices"]))
    # flat-vol LV MC must price the flat BS surface (loose tol: 2k paths)
    iv = mc_implied_vols(
        np.asarray(a["prices"]),
        spot=SPOT,
        r=R,
        q=Q,
        strikes=CALIB_STRIKES,
        maturities=surf.maturities,
    )
    assert np.abs(iv - 0.25).max() < 0.03

    with pytest.raises(ValueError, match="scheme"):
        local_vol_mc(surf, seed=1, scheme="runge-kutta", **kw)
    with pytest.raises(ValueError, match="stream_id"):
        local_vol_mc(surf, seed=1, stream_id=3, **kw)
    with pytest.raises(ValueError):
        local_vol_mc(surf, seed=1, **{**kw, "n_paths": 2_001})
    with pytest.raises(ValueError):
        local_vol_mc("not-a-surface", seed=1, **kw)  # type: ignore[arg-type]


@pytest.mark.synthetic
def test_local_vol_mc_milstein_close_to_euler() -> None:
    """Milstein correction is small on a smooth extracted surface (same seed)."""
    surf = dupire_local_vol(STRIKES, MATURITIES, _sabr_surface(), spot=SPOT, r=R, q=Q)
    common = dict(spot=SPOT, r=R, q=Q, strikes=STRIKES, seed=13, n_paths=20_000, max_dt=1.0 / 24.0)
    iv_e = mc_implied_vols(
        np.asarray(local_vol_mc(surf, scheme="euler", **common)["prices"]),
        spot=SPOT,
        r=R,
        q=Q,
        strikes=STRIKES,
        maturities=MATURITIES,
    )
    iv_m = mc_implied_vols(
        np.asarray(local_vol_mc(surf, scheme="milstein", **common)["prices"]),
        spot=SPOT,
        r=R,
        q=Q,
        strikes=STRIKES,
        maturities=MATURITIES,
    )
    band = _moneyness_band()
    assert np.abs(iv_e - iv_m)[band].max() < 0.02
    assert np.abs(iv_m - _sabr_iv_grid())[band].max() < 0.02


def test_heston_mc_determinism_and_fail_closed() -> None:
    times = np.linspace(0.0, 1.0, 21)
    base = dict(spot=SPOT, r=R, times=times, n_paths=2_000, seed=3, **HESTON)
    a = heston_mc(**base)
    b = heston_mc(**base)
    assert np.array_equal(np.asarray(a["S"]), np.asarray(b["S"]))
    assert np.array_equal(np.asarray(a["v"]), np.asarray(b["v"]))
    c = heston_mc(**{**base, "seed": 4})
    assert not np.array_equal(np.asarray(a["S"]), np.asarray(c["S"]))
    assert a["synthetic"] is True

    with pytest.raises(ValueError):
        heston_mc(**{**base, "rho": 1.0})
    with pytest.raises(ValueError):
        heston_mc(**{**base, "xi": -0.1})
    with pytest.raises(ValueError):
        heston_mc(**{**base, "v0": 0.0})
    with pytest.raises(ValueError):
        heston_mc(**{**base, "kappa": 0.0})
    with pytest.raises(ValueError):
        heston_mc(**{**base, "times": np.linspace(0.1, 1.0, 21)})
    with pytest.raises(ValueError):
        heston_mc(**{**base, "n_paths": 2_001})
    with pytest.raises(ValueError):
        heston_mc(**{**base, "seed": -1})


@pytest.mark.synthetic
def test_slv_leverage_convergence_matched_heston() -> None:
    """van der Stoep et al. (2014) mixing MC: matched target => L* == 1 fixed
    point, implied-vol error against the closed-form Heston target shrinks and
    the leverage update contracts."""
    target_iv = np.array(
        [heston_implied_vol(k, 1.0, S0=SPOT, r=R, q=Q, **HESTON) for k in CALIB_STRIKES]
    )
    out = slv_leverage_function(
        spot=SPOT,
        r=R,
        q=Q,
        T=1.0,
        calib_strikes=CALIB_STRIKES,
        target_implied_vols=target_iv,
        n_steps=40,
        n_paths=12_000,
        n_iter=3,
        seed=5,
        n_s_bins=12,
        n_time_nodes=8,
        **HESTON,
    )
    its = out["iterations"]
    assert [it["iter"] for it in its] == [0.0, 1.0, 2.0, 3.0]  # type: ignore[index]
    assert its[0]["implied_rmse"] < 0.01  # L==1 baseline already near the CF target
    assert its[-1]["implied_rmse"] < 0.01
    assert its[-1]["implied_rmse"] <= its[0]["implied_rmse"] + 0.003
    assert its[-1]["l2_delta_rms"] < its[1]["l2_delta_rms"]  # contraction to fixed point
    lev = np.asarray(out["leverage"])
    assert np.isfinite(lev).all()
    assert lev.min() > 0.5 and lev.max() < 1.5
    assert 0.9 <= float(lev.mean()) <= 1.1  # fixed point L == 1 up to binning noise
    assert its[-1]["bins_updated_frac"] >= 0.5
    assert np.abs(np.asarray(out["final_implied_vols"]) - target_iv).max() < 0.01
    assert out["synthetic"] is True
    assert "SYNTHETIC" in str(out["label"])
    # the calibrated grid composes into a usable leverage callable
    fn = make_leverage_interpolator(
        np.asarray(out["time_nodes"]), np.asarray(out["bin_centers"]), lev
    )
    vals = fn(np.array([SPOT, 1.2 * SPOT]), 0.5)
    assert vals.shape == (2,)
    assert np.isfinite(vals).all() and (vals > 0.0).all()


@pytest.mark.synthetic
def test_slv_determinism_same_seed() -> None:
    target_iv = np.array(
        [heston_implied_vol(k, 1.0, S0=SPOT, r=R, q=Q, **HESTON) for k in CALIB_STRIKES[:3]]
    )
    kw = dict(
        spot=SPOT,
        r=R,
        q=Q,
        T=1.0,
        calib_strikes=CALIB_STRIKES[:3],
        target_implied_vols=target_iv,
        n_steps=20,
        n_paths=4_000,
        n_iter=2,
        seed=2,
        n_s_bins=8,
        n_time_nodes=4,
        **HESTON,
    )
    a = slv_leverage_function(**kw)
    b = slv_leverage_function(**kw)
    assert np.array_equal(np.asarray(a["leverage"]), np.asarray(b["leverage"]))
    assert np.array_equal(np.asarray(a["final_implied_vols"]), np.asarray(b["final_implied_vols"]))


@pytest.mark.synthetic
def test_slv_zero_vov_leverage_stays_one() -> None:
    """Zero vol-of-vol: target == driver == deterministic variance, so the
    mixing-MC fixed point is exactly L == 1 (degenerate consistency)."""
    hp0 = dict(HESTON, xi=0.0, v0=0.09)
    times = np.linspace(0.0, 1.0, 21)
    sim = heston_mc(spot=SPOT, r=R, q=Q, times=times, n_paths=1_000, seed=4, **hp0)
    v = np.asarray(sim["v"])[0]
    sig_bs = float(np.sqrt(np.sum(v[:-1]) * (times[1] - times[0])))
    out = slv_leverage_function(
        spot=SPOT,
        r=R,
        q=Q,
        T=1.0,
        calib_strikes=CALIB_STRIKES[:3],
        target_implied_vols=np.full(3, sig_bs),
        n_steps=20,
        n_paths=4_000,
        n_iter=2,
        seed=1,
        n_s_bins=8,
        n_time_nodes=4,
        **hp0,
    )
    assert np.abs(np.asarray(out["leverage"]) - 1.0).max() < 1e-12
    assert out["final_implied_rmse"] < 0.005


@pytest.mark.synthetic
def test_zero_vov_slv_matches_closed_form_and_local_vol_mc() -> None:
    """Degenerate limits: with xi == 0 and leverage == 1 the SLV MC is a pure
    (deterministic) local-vol model — it must match the closed-form Black price
    with the scheme-exact variance, and match local_vol_mc on sigma_L = sqrt(v(t))."""
    hp0 = dict(HESTON, xi=0.0, v0=0.09)
    times = np.linspace(0.0, 1.0, 41)
    dt = float(times[1] - times[0])
    sim = heston_mc(spot=SPOT, r=R, q=0.0, times=times, n_paths=20_000, seed=9, **hp0)
    v_all = np.asarray(sim["v"])
    assert np.allclose(v_all, v_all[0])  # deterministic variance across paths
    v = v_all[0]

    sig_slv = float(np.sqrt(np.sum(v[:-1]) * dt))
    p_bs = np.array([bs_price(SPOT, k, 1.0, sig_slv, R) for k in CALIB_STRIKES])
    mc_none = slv_price_calls(
        spot=SPOT,
        r=R,
        q=0.0,
        T=1.0,
        strikes=CALIB_STRIKES,
        n_paths=20_000,
        n_steps=40,
        seed=9,
        **hp0,
    )
    ones = make_leverage_interpolator(times[1:], np.array([50.0, 100.0, 200.0]), np.ones((40, 3)))
    mc_lev1 = slv_price_calls(
        spot=SPOT,
        r=R,
        q=0.0,
        T=1.0,
        strikes=CALIB_STRIKES,
        n_paths=20_000,
        n_steps=40,
        seed=9,
        leverage=ones,
        **hp0,
    )
    # leverage == 1 exactly recovers the un-levered (pure local-vol) simulation
    assert np.array_equal(np.asarray(mc_none["prices"]), np.asarray(mc_lev1["prices"]))
    se = np.asarray(mc_none["stderr"])
    assert np.all(np.abs(np.asarray(mc_none["prices"]) - p_bs) <= 3.0 * se + 0.01)

    # local_vol_mc on the same deterministic variance reproduces the same prices.
    # Step 0 of the LV MC clips t=0 to the first maturity node, so the
    # scheme-exact variance is dt*(2*v1 + sum v2..v39).
    kgrid = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    sig_lv = float(np.sqrt((2.0 * v[1] + np.sum(v[2:40])) * dt))
    surf = LocalVolSurface(kgrid, times[1:], np.tile(np.sqrt(v[1:])[:, None], (1, kgrid.size)))
    mc_lv = local_vol_mc(
        surf,
        spot=SPOT,
        r=R,
        q=0.0,
        strikes=CALIB_STRIKES,
        seed=9,
        n_paths=20_000,
        max_dt=dt,
    )
    p_lv_bs = np.array([bs_price(SPOT, k, 1.0, sig_lv, R) for k in CALIB_STRIKES])
    se_lv = np.asarray(mc_lv["stderr"])[-1]
    assert np.all(np.abs(np.asarray(mc_lv["prices"])[-1] - p_lv_bs) <= 3.0 * se_lv + 0.01)
    ivs = np.array(
        [
            implied_vol(p, SPOT, k, 1.0, R)
            for p, k in zip(np.asarray(mc_lv["prices"])[-1], CALIB_STRIKES, strict=True)
        ]
    )
    assert np.abs(ivs - sig_lv).max() < 0.005


def test_make_leverage_interpolator_fail_closed() -> None:
    tn = np.array([0.25, 0.5, 1.0])
    bc = np.array([80.0, 100.0, 120.0])
    with pytest.raises(ValueError):
        make_leverage_interpolator(tn, bc, np.ones((2, 3)))
    with pytest.raises(ValueError):
        make_leverage_interpolator(tn, bc, np.zeros((3, 3)))
    fn = make_leverage_interpolator(tn, bc, np.ones((3, 3)) * 1.3)
    assert fn(np.array([SPOT]), 0.5) == pytest.approx(1.3)
    # clamped extrapolation outside the box
    assert fn(np.array([1e6]), 5.0) == pytest.approx(1.3)
    with pytest.raises(ValueError):
        fn(np.array([-1.0]), 0.5)


def test_slv_leverage_function_fail_closed() -> None:
    """Invalid SLV/target parameters raise before any simulation runs."""
    base = dict(
        spot=SPOT,
        r=R,
        q=Q,
        T=1.0,
        calib_strikes=CALIB_STRIKES[:3],
        target_implied_vols=np.full(3, 0.2),
        n_steps=8,
        n_paths=200,
        n_iter=1,
        seed=0,
        n_s_bins=4,
        n_time_nodes=3,
        **HESTON,
    )
    with pytest.raises(ValueError):
        slv_leverage_function(**{**base, "rho": 1.0})
    with pytest.raises(ValueError):
        slv_leverage_function(**{**base, "xi": -0.5})
    with pytest.raises(ValueError):
        slv_leverage_function(**{**base, "n_iter": 0})
    with pytest.raises(ValueError):
        slv_leverage_function(**{**base, "damping": 0.0})
    with pytest.raises(ValueError):
        slv_leverage_function(**{**base, "l2_min": 1.5})
    with pytest.raises(ValueError):
        slv_leverage_function(**{**base, "target_implied_vols": np.full(2, 0.2)})
    with pytest.raises(ValueError):
        slv_leverage_function(**{**base, "target_rho": 1.0})
