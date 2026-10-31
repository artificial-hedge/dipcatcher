"""Tests for quant_fund.models.ivs_diffusion — AD-Seq-Vol (Han et al. 2026).

References: Han, Zhang, Torres, Acero & Xu (2026, arXiv:2609.13402);
Ho, Jain & Abbeel (2020, DDPM); Gatheral & Jacquier (2014, arXiv:1204.0646,
arb-free SVI); Gneiting & Raftery (2007, energy score); Rockafellar & Uryasev
(2000/2002, ES/CVaR); Black & Scholes (1973).

All data here is SYNTHETIC (seeded SVI-with-leverage stream) — correctness
evidence for the algorithm, never market evidence; no live-trading claims.
Torch tests skip cleanly when the nn extra is absent.
"""

from __future__ import annotations

import importlib.util
import math
import sys

import numpy as np
import pytest

from quant_fund.models import ivs_diffusion as ivd
from quant_fund.quant_models.black_scholes import bs_price


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="ivs diffusion training requires the nn extra (torch)"
)

GRID = ivd.IVSGrid(
    log_moneyness=np.linspace(-0.35, 0.25, 7),
    maturities=np.array([0.10, 0.25, 0.50, 1.0]),
)


def _stream(n: int = 220, seed: int = 0, jitter: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    return ivd.synthetic_ivs_stream(n, GRID, seed=seed, jitter=jitter)


# ---------------------------------------------------------------------------
# IVSGrid and fail-closed validation (numpy only)
# ---------------------------------------------------------------------------


def test_grid_properties() -> None:
    assert GRID.n_k == 7 and GRID.n_tau == 4 and GRID.n_points == 28
    k = GRID.log_moneyness
    assert np.allclose(GRID.strikes, np.exp(k))
    assert np.all(np.diff(GRID.strikes) > 0.0)


def test_grid_fail_closed() -> None:
    with pytest.raises(ValueError, match=">= 3 grid points"):
        ivd.IVSGrid(np.array([-0.2, 0.0]), np.array([0.1, 1.0]))
    with pytest.raises(ValueError, match=">= 2 tenors"):
        ivd.IVSGrid(np.array([-0.2, 0.0, 0.2]), np.array([0.5]))
    with pytest.raises(ValueError, match="finite"):
        ivd.IVSGrid(np.array([-0.2, np.nan, 0.2]), np.array([0.1, 1.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        ivd.IVSGrid(np.array([-0.2, 0.3, 0.1]), np.array([0.1, 1.0]))
    with pytest.raises(ValueError, match="positive"):
        ivd.IVSGrid(np.array([-0.2, 0.0, 0.2]), np.array([0.0, 1.0]))


def test_check_surfaces_promotion_and_fail_closed() -> None:
    one = np.full((7, 4), 0.2)
    assert GRID.check_surfaces(one, positive=True, name="v").shape == (1, 7, 4)
    batch = np.full((3, 7, 4), 0.2)
    assert GRID.check_surfaces(batch, positive=True, name="v").shape == (3, 7, 4)
    with pytest.raises(ValueError, match="does not match grid"):
        GRID.check_surfaces(np.full((2, 5, 4), 0.2), positive=True, name="v")
    with pytest.raises(ValueError, match="finite"):
        GRID.check_surfaces(np.full((1, 7, 4), np.nan), positive=False, name="v")
    with pytest.raises(ValueError, match="strictly positive"):
        GRID.check_surfaces(np.full((1, 7, 4), -0.1), positive=True, name="v")
    # non-positive entries are allowed under positive=False (audit path)
    out = GRID.check_surfaces(np.zeros((1, 7, 4)), positive=False, name="v")
    assert out.shape == (1, 7, 4)


# ---------------------------------------------------------------------------
# call-price surface and the arb audit (numpy only)
# ---------------------------------------------------------------------------


def test_call_price_surface_matches_bs_primitive() -> None:
    v = np.full((2, 7, 4), 0.3)
    prices = ivd.call_price_surface(v, GRID)
    assert prices.shape == (2, 7, 4)
    k, t = GRID.log_moneyness[:, None], GRID.maturities[None, :]
    ref = bs_price(1.0, np.exp(k), np.broadcast_to(t, (7, 4)), 0.0, 0.0, 0.3, "call")
    assert np.allclose(prices, np.broadcast_to(ref, (2, 7, 4)))
    # bounds: 0 <= C <= 1 (S=1, r=q=0), decreasing in k, increasing in tau
    assert np.all(prices >= 0.0) and np.all(prices <= 1.0)
    assert np.all(np.diff(prices, axis=1) < 0.0)
    assert np.all(np.diff(prices, axis=2) > 0.0)


def test_butterfly_residuals_nonnegative_on_flat_surface() -> None:
    # a flat-vol surface is arbitrage-free: all second divided differences >= 0
    prices = ivd.call_price_surface(np.full((4, 7, 4), 0.25), GRID)
    dd = ivd._butterfly_residuals(prices, GRID.strikes)
    assert dd.shape == (4, 5, 4)
    assert np.all(dd > -1e-9)


def test_arb_report_clean_stream_is_zero() -> None:
    _, v = _stream(60, seed=1, jitter=0.0)
    rep = ivd.arb_violation_report(v, GRID)
    assert rep.n_surfaces == 60
    assert rep.surface_rate == 0.0
    assert rep.nonneg_rate == 0.0
    assert rep.calendar_rate == 0.0
    assert rep.butterfly_rate == 0.0
    d = rep.as_dict(prefix="arb_")
    assert d["arb_surface_rate"] == 0.0 and "arb_butterfly_rate" in d


def test_arb_report_detects_nonnegative_violation() -> None:
    v = np.full((2, 7, 4), 0.25)
    v[0, 3, 1] = -0.05  # one negative-vol cell
    rep = ivd.arb_violation_report(v, GRID)
    assert rep.nonneg_rate == pytest.approx(1.0 / 56.0)
    assert rep.nonneg_magnitude == pytest.approx(0.05)
    assert rep.surface_rate == 0.5


def test_arb_report_detects_calendar_violation() -> None:
    v = np.full((2, 7, 4), 0.25)
    v[:, :, 3] = 0.10  # last maturity crushed -> w falls in tau
    rep = ivd.arb_violation_report(v, GRID)
    assert rep.calendar_rate == pytest.approx(1.0 / 3.0)  # 1 of 3 tau-pairs
    assert rep.calendar_magnitude > 0.0
    assert rep.surface_rate == 1.0


def test_arb_report_detects_butterfly_violation() -> None:
    v = np.full((4, 7, 4), 0.25)
    v[:, 3, :] = 0.55  # spike at the middle strike breaks convexity
    rep = ivd.arb_violation_report(v, GRID)
    assert rep.butterfly_rate > 0.0
    assert rep.butterfly_magnitude > 0.0
    assert rep.surface_rate == 1.0


def test_arb_report_fail_closed() -> None:
    with pytest.raises(ValueError):
        ivd.arb_violation_report(np.zeros((0, 7, 4)), GRID)
    with pytest.raises(ValueError):
        ivd.arb_violation_report(np.full((1, 7, 4), np.inf), GRID)


# ---------------------------------------------------------------------------
# synthetic stream DGP (numpy only)
# ---------------------------------------------------------------------------


def test_stream_shapes_positive_and_deterministic() -> None:
    r1, v1 = _stream(40, seed=7)
    r2, v2 = _stream(40, seed=7)
    r3, _ = _stream(40, seed=8)
    assert r1.shape == (40,) and v1.shape == (40, 7, 4)
    assert np.all(v1 > 0.0) and np.all(np.isfinite(v1))
    assert np.array_equal(r1, r2) and np.array_equal(v1, v2)
    assert not np.array_equal(r1, r3)


def test_stream_leverage_sign() -> None:
    # negative leverage: returns correlate negatively with level innovations,
    # i.e. with the mean surface increment
    r, v = _stream(400, seed=3, jitter=0.0)
    dv = np.diff(v.reshape(len(v), -1), axis=0).mean(axis=1)
    corr = np.corrcoef(r[1:], dv)[0, 1]
    assert corr < -0.2


def test_stream_jitter_creates_violations_but_stays_positive() -> None:
    _, v = _stream(120, seed=5, jitter=0.10)
    assert np.all(v > 0.0)  # multiplicative jitter never goes non-positive
    rep = ivd.arb_violation_report(v, GRID)
    assert rep.surface_rate > 0.5


def test_stream_fail_closed_params() -> None:
    with pytest.raises(ValueError, match="n_steps"):
        ivd.synthetic_ivs_stream(0, GRID)
    with pytest.raises(ValueError, match="jitter"):
        ivd.synthetic_ivs_stream(10, GRID, jitter=-1.0)
    with pytest.raises(ValueError, match="momentum"):
        ivd.synthetic_ivs_stream(10, GRID, momentum=1.0)
    with pytest.raises(ValueError, match="leverage"):
        ivd.synthetic_ivs_stream(10, GRID, leverage=1.5)
    with pytest.raises(ValueError, match="atm_level"):
        ivd.synthetic_ivs_stream(10, GRID, atm_level=0.0)


# ---------------------------------------------------------------------------
# smile codes and supervised pairs (numpy only)
# ---------------------------------------------------------------------------


def test_surface_codes_recover_quadratic_rows() -> None:
    # a surface row that is exactly quadratic in k yields (a, b, c) codes
    k = GRID.log_moneyness
    v = np.empty((2, 7, 4))
    for j in range(4):
        a, b, c = 0.2 + 0.05 * j, -0.1 + 0.02 * j, 0.15 + 0.01 * j
        v[:, :, j] = a + b * k + c * k * k
    codes = ivd.surface_codes(v, GRID)
    assert codes.shape == (2, 12)
    for j in range(4):
        assert codes[0, 3 * j] == pytest.approx(0.2 + 0.05 * j, abs=1e-10)
        assert codes[0, 3 * j + 1] == pytest.approx(-0.1 + 0.02 * j, abs=1e-10)
        assert codes[0, 3 * j + 2] == pytest.approx(0.15 + 0.01 * j, abs=1e-10)


def test_surface_codes_linear_and_fail_closed() -> None:
    _, v = _stream(8, seed=2)
    a, b = ivd.surface_codes(v, GRID), ivd.surface_codes(2.0 * v, GRID)
    assert np.allclose(b, 2.0 * a)
    with pytest.raises(ValueError):
        ivd.surface_codes(np.full((1, 3, 4), 0.2), GRID)


def test_build_pairs_shapes_target_and_context_layout() -> None:
    r, v = _stream(60, seed=4)
    h = 6
    X, y, v_last = ivd.build_supervised_pairs(r, v, GRID, h)
    n_pairs = 60 - h
    assert X.shape == (n_pairs, h * 13)  # h returns + 1 level + (h-1) incrs of 12
    assert y.shape == (n_pairs, 29) and v_last.shape == (n_pairs, 7, 4)
    # first pair: context covers t = h-1, target is step h
    i = 0
    t = h - 1
    assert y[i, 0] == r[t + 1]
    assert np.allclose(y[i, 1:], (v[t + 1] - v[t]).reshape(-1))
    assert np.array_equal(v_last[i], v[t])
    codes = ivd.surface_codes(v, GRID)
    dc = codes[1:] - codes[:-1]
    expect = np.concatenate([r[0:h], codes[t], dc[0:t].reshape(-1)])
    assert np.allclose(X[i], expect)


def test_build_pairs_fail_closed() -> None:
    r, v = _stream(20, seed=0)
    with pytest.raises(ValueError, match="no supervised pairs"):
        ivd.build_supervised_pairs(r, v, GRID, 25)
    with pytest.raises(ValueError, match="equal length"):
        ivd.build_supervised_pairs(r[:10], v, GRID, 4)
    with pytest.raises(ValueError, match="window"):
        ivd.build_supervised_pairs(r, v, GRID, 0)


# ---------------------------------------------------------------------------
# resample baseline and proper scoring (numpy only)
# ---------------------------------------------------------------------------


def test_resample_shapes_anchor_and_determinism() -> None:
    r, v = _stream(120, seed=1)
    h, hz = 6, 3
    rp1, vp1 = ivd.historical_resample_scenarios(r, v, GRID, h, n_scenarios=10, horizon=hz, seed=9)
    rp2, vp2 = ivd.historical_resample_scenarios(r, v, GRID, h, n_scenarios=10, horizon=hz, seed=9)
    assert rp1.shape == (10, hz) and vp1.shape == (10, hz, 7, 4)
    assert np.array_equal(rp1, rp2) and np.array_equal(vp1, vp2)
    # block resample anchored at each block's window-end: every first-step
    # scenario surface is exactly a realized historical surface
    for i in range(10):
        matches = np.all(np.isclose(vp1[i, 0], v), axis=(1, 2))
        assert bool(matches.any())
    assert np.all(np.isfinite(vp1))


def test_resample_fail_closed() -> None:
    r, v = _stream(30, seed=0)
    with pytest.raises(ValueError, match="window \\+ horizon"):
        ivd.historical_resample_scenarios(r, v, GRID, 20, n_scenarios=4, horizon=20)
    with pytest.raises(ValueError, match="n_scenarios"):
        ivd.historical_resample_scenarios(r, v, GRID, 5, n_scenarios=0, horizon=3)


def test_scenario_energy_score_is_proper_wrapper() -> None:
    rng = np.random.default_rng(0)
    ens = rng.standard_normal((64, 5))
    y = np.zeros(5)
    good = ivd.scenario_energy_score(np.broadcast_to(y, (8, 5)).copy(), y)
    bad = ivd.scenario_energy_score(ens + 5.0, y)
    assert good < bad


# ---------------------------------------------------------------------------
# optimization-based hedge evaluation (numpy only)
# ---------------------------------------------------------------------------


def _scenarios(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    r, v = _stream(n + 2, seed=seed)
    return r[2:], v[2:]


def test_hedge_delta_solves_normal_equation() -> None:
    r_s, v_s = _scenarios(64, seed=3)
    out = ivd.evaluate_hedge(r_s, v_s, v_s[1:2], GRID, k_idx=3, tau_idx=2)
    # least-squares optimality: residual orthogonal to dS
    s1 = np.exp(r_s) - 1.0
    kk, tt = float(GRID.log_moneyness[3]), float(GRID.maturities[2])
    c0 = float(bs_price(1.0, math.exp(kk), tt, 0.0, 0.0, v_s[1, 3, 2], "call"))
    c1 = bs_price(np.exp(r_s), math.exp(kk), tt, 0.0, 0.0, v_s[:, 3, 2], "call")
    resid = np.asarray(c1 - c0) - out["hedge_delta"] * s1
    assert abs(float(np.dot(resid, s1))) < 1e-6
    assert out["hedge_tracking_rmse"] <= out["unhedged_tracking_rmse"] + 1e-9
    assert 0.0 < out["hedge_bs_delta"] < 1.0


def test_hedge_fail_closed() -> None:
    r_s, v_s = _scenarios(16, seed=3)
    base = v_s[1:2]
    with pytest.raises(ValueError, match="identical"):
        ivd.evaluate_hedge(np.zeros(16), v_s, base, GRID, k_idx=3, tau_idx=2)
    with pytest.raises(ValueError, match="alpha"):
        ivd.evaluate_hedge(r_s, v_s, base, GRID, k_idx=3, tau_idx=2, alpha=1.5)
    with pytest.raises(ValueError, match="k_idx"):
        ivd.evaluate_hedge(r_s, v_s, base, GRID, k_idx=9, tau_idx=2)
    with pytest.raises(ValueError, match="tau_idx"):
        ivd.evaluate_hedge(r_s, v_s, base, GRID, k_idx=3, tau_idx=7)
    with pytest.raises(ValueError, match=">= 4 scenarios"):
        ivd.evaluate_hedge(r_s[:3], v_s[:3], base, GRID, k_idx=3, tau_idx=2)
    with pytest.raises(ValueError, match="single surface"):
        ivd.evaluate_hedge(r_s, v_s, v_s[:2], GRID, k_idx=3, tau_idx=2)


# ---------------------------------------------------------------------------
# torch gating: module must import and fail closed without the nn extra
# ---------------------------------------------------------------------------


def test_module_imports_without_torch_and_raises_clear_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No top-level torch import; fit raises a guided ImportError."""
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    spec = importlib.util.spec_from_file_location("_ivd_no_torch", ivd.__file__)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_ivd_no_torch", probe)
    spec.loader.exec_module(probe)  # must import cleanly without torch
    # numpy core stays usable while torch is blocked
    r, v = probe.synthetic_ivs_stream(20, GRID, seed=0)
    assert v.shape == (20, 7, 4)
    model = probe.IVSDiffusion(n_steps=4, epochs=1, hidden=(8,))
    with pytest.raises(ImportError, match=r"'nn' extra"):
        model.fit(r, v, GRID, window=4)


# ---------------------------------------------------------------------------
# torch lane (skipped when the nn extra is absent) — tiny budgets only
# ---------------------------------------------------------------------------


def _fitted_model(
    n: int = 200, seed: int = 2, epochs: int = 60
) -> tuple[ivd.IVSDiffusion, np.ndarray, np.ndarray]:
    r, v = ivd.synthetic_ivs_stream(n, GRID, seed=seed, jitter=0.03, momentum=0.8)
    model = ivd.IVSDiffusion(
        n_steps=8, epochs=epochs, hidden=(32, 32), batch_size=32, seed=seed
    ).fit(r[: n - 30], v[: n - 30], GRID, window=6)
    return model, r, v


@requires_torch
def test_fit_sets_state_and_loss_decreases() -> None:
    model, _, _ = _fitted_model(epochs=50)
    assert model.is_fitted
    assert model.fit_info is not None
    curve = model.fit_info.loss_curve
    assert len(curve) > 0 and all(np.isfinite(curve))
    assert float(np.mean(curve[-5:])) < float(np.mean(curve[:5]))


@requires_torch
def test_fit_fail_closed() -> None:
    r, v = _stream(30, seed=0)
    model = ivd.IVSDiffusion(n_steps=4, epochs=2, hidden=(8,), seed=0)
    with pytest.raises(ValueError, match="stream too short"):
        model.fit(r[:10], v[:10], GRID, window=6)
    with pytest.raises(ValueError, match="equal length"):
        model.fit(r[:20], v[:19], GRID, window=4)
    v_neg = v.copy()
    v_neg[5, 0, 0] = -0.01
    with pytest.raises(ValueError, match="strictly positive"):
        model.fit(r, v_neg, GRID, window=4)
    with pytest.raises(ValueError, match="n_steps"):
        ivd.IVSDiffusion(n_steps=1)
    with pytest.raises(ValueError, match="schedule"):
        ivd.IVSDiffusion(schedule="nope")


@requires_torch
def test_unfitted_calls_raise() -> None:
    r, v = _stream(20, seed=0)
    model = ivd.IVSDiffusion(n_steps=4, epochs=2, hidden=(8,), seed=0)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.sample_next(r[-6:], v[-6:], n_samples=4, seed=0)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.generate_scenarios(r[-6:], v[-6:], n_scenarios=4, horizon=2, seed=0)


@requires_torch
def test_sample_next_shapes_and_surface_reconstruction() -> None:
    model, r, v = _fitted_model()
    rn, dv, vn = model.sample_next(r[-6:], v[-6:], n_samples=16, seed=3)
    assert rn.shape == (16,) and dv.shape == (16, 7, 4) and vn.shape == (16, 7, 4)
    assert np.allclose(vn, v[-1][None, :, :] + dv)


@requires_torch
def test_sample_next_deterministic_and_seed_sensitive() -> None:
    model, r, v = _fitted_model()
    a = model.sample_next(r[-6:], v[-6:], n_samples=8, seed=11)
    b = model.sample_next(r[-6:], v[-6:], n_samples=8, seed=11)
    c = model.sample_next(r[-6:], v[-6:], n_samples=8, seed=12)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)
    assert not np.array_equal(a[2], c[2])


@requires_torch
def test_sample_next_fail_closed_window_length() -> None:
    model, r, v = _fitted_model()
    with pytest.raises(ValueError, match="window=6"):
        model.sample_next(r[-4:], v[-4:], n_samples=4, seed=0)
    with pytest.raises(ValueError, match="n_samples"):
        model.sample_next(r[-6:], v[-6:], n_samples=0, seed=0)


@requires_torch
def test_generate_scenarios_adapted_shapes_and_determinism() -> None:
    model, r, v = _fitted_model()
    rp, vp = model.generate_scenarios(r[-6:], v[-6:], n_scenarios=12, horizon=4, seed=5)
    assert rp.shape == (12, 4) and vp.shape == (12, 4, 7, 4)
    assert np.all(np.isfinite(vp))
    rp2, vp2 = model.generate_scenarios(r[-6:], v[-6:], n_scenarios=12, horizon=4, seed=5)
    assert np.array_equal(rp, rp2) and np.array_equal(vp, vp2)
    # scenario paths are increment chains anchored at the last realized surface
    assert not np.allclose(vp[:, 0], np.broadcast_to(v[-1], vp[:, 0].shape))


@requires_torch
def test_generate_scenarios_fail_closed() -> None:
    model, r, v = _fitted_model()
    with pytest.raises(ValueError, match="n_scenarios"):
        model.generate_scenarios(r[-6:], v[-6:], n_scenarios=0, horizon=2, seed=0)
    with pytest.raises(ValueError, match="horizon"):
        model.generate_scenarios(r[-6:], v[-6:], n_scenarios=4, horizon=0, seed=0)
    with pytest.raises(ValueError, match="window=6"):
        model.generate_scenarios(r[-3:], v[-3:], n_scenarios=4, horizon=2, seed=0)


@requires_torch
def test_conditional_targets_shape_and_determinism() -> None:
    model, r, v = _fitted_model()
    s1 = model.conditional_targets(r[-6:], v[-6:], n_samples=10, seed=4)
    s2 = model.conditional_targets(r[-6:], v[-6:], n_samples=10, seed=4)
    assert s1.shape == (10, 29)
    assert np.array_equal(s1, s2)


@requires_torch
def test_model_learns_conditional_structure() -> None:
    """Model beats the unconditional block-resample baseline on energy score.

    SYNTHETIC correctness check: on the momentum DGP the next-step surface
    increment is genuinely predictable, so a trained conditional model must
    outscore unconditional resampling of historical increments.
    """
    r, v = ivd.synthetic_ivs_stream(320, GRID, seed=1, jitter=0.02, momentum=0.8)
    model = ivd.IVSDiffusion(n_steps=10, epochs=250, hidden=(64, 64), batch_size=64, seed=1).fit(
        r[:260], v[:260], GRID, window=8
    )
    X_all, y_all, _ = ivd.build_supervised_pairs(r, v, GRID, 8)
    params = model._require_params()
    ev = slice(252, X_all.shape[0])
    ctxs = np.asarray((X_all[ev] - params.ctx_mean) / params.ctx_std, dtype=float)
    y_ev = np.asarray((y_all[ev] - params.y_mean) / params.y_std, dtype=float)
    y_tr = np.asarray(
        (ivd.build_supervised_pairs(r[:260], v[:260], GRID, 8)[1] - params.y_mean) / params.y_std,
        dtype=float,
    )
    rng_m = np.random.default_rng(7)
    rng_r = np.random.default_rng(3)
    es_m, es_r = [], []
    for i in range(ctxs.shape[0]):
        ens = ivd._np_sample(params, np.repeat(ctxs[i : i + 1], 24, axis=0), rng_m)
        res = y_tr[rng_r.integers(0, len(y_tr), 24)]
        es_m.append(ivd.scenario_energy_score(ens, y_ev[i]))
        es_r.append(ivd.scenario_energy_score(res, y_ev[i]))
    assert float(np.mean(es_m)) < float(np.mean(es_r))


@requires_torch
def test_finetune_reduces_violations() -> None:
    """AD-Seq-Vol-FT: post-training arb penalty cuts the probed violation rate."""
    r, v = ivd.synthetic_ivs_stream(200, GRID, seed=2, jitter=0.03, momentum=0.8)
    model = ivd.IVSDiffusion(n_steps=8, epochs=60, hidden=(32, 32), batch_size=32, seed=2).fit(
        r[:170], v[:170], GRID, window=6
    )
    ft = model.finetune_no_arb(
        r[:170], v[:170], penalty_weight=30.0, epochs=20, lr=2e-3, n_probe=64, seed=4
    )
    assert ft.penalty_curve[-1] < ft.penalty_curve[0]
    assert ft.cell_rate_after < ft.cell_rate_before
    assert ft.violation_rate_after <= ft.violation_rate_before
    assert model.finetune_info is ft
    # model still functional after finetune
    rn, dv, vn = model.sample_next(r[-6:], v[-6:], n_samples=4, seed=1)
    assert vn.shape == (4, 7, 4) and np.all(np.isfinite(vn))


@requires_torch
def test_finetune_fail_closed() -> None:
    r, v = _stream(60, seed=0)
    model = ivd.IVSDiffusion(n_steps=4, epochs=3, hidden=(8,), seed=0)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.finetune_no_arb(r, v, penalty_weight=1.0, epochs=2)
    model.fit(r, v, GRID, window=5)
    with pytest.raises(ValueError, match="penalty_weight"):
        model.finetune_no_arb(r, v, penalty_weight=0.0, epochs=2)
    with pytest.raises(ValueError, match="epochs"):
        model.finetune_no_arb(r, v, penalty_weight=1.0, epochs=0)


@requires_torch
def test_end_to_end_determinism() -> None:
    """Same seed -> bit-identical fitted model and generated paths."""
    m1, r, v = _fitted_model(n=160, seed=9, epochs=40)
    m2, _, _ = _fitted_model(n=160, seed=9, epochs=40)
    p1 = m1.generate_scenarios(r[-6:], v[-6:], n_scenarios=6, horizon=3, seed=2)
    p2 = m2.generate_scenarios(r[-6:], v[-6:], n_scenarios=6, horizon=3, seed=2)
    assert np.array_equal(p1[0], p2[0]) and np.array_equal(p1[1], p2[1])


@requires_torch
def test_bench_smoke_keys_labels_and_ft_direction() -> None:
    out = ivd.bench_ivs_diffusion(
        n_stream=240,
        window=6,
        horizon=3,
        n_scenarios=20,
        n_train=200,
        n_steps=8,
        epochs=120,
        ft_epochs=15,
        penalty_weight=30.0,
        hidden=(48, 48),
        jitter=0.03,
        batch_size=64,
    )
    assert out["synthetic_claim"] == "research_metric_only"
    assert out["synthetic"] == "seeded_ivs_stream"
    for key in (
        "synthetic_es_model",
        "synthetic_es_resample",
        "synthetic_hedge_es_tail_model",
        "synthetic_hedge_rmse_model",
    ):
        assert math.isfinite(float(out[key]))
    # FT direction at this budget: cell-level violations strictly reduced
    assert float(out["synthetic_ft_probe_cell_after"]) < float(
        out["synthetic_ft_probe_cell_before"]
    )
    assert float(out["synthetic_arb_cell_finetuned"]) < float(out["synthetic_arb_cell_generated"])
    # no forbidden headline metrics leak into the report
    banned = ("sharpe", "sortino", "calmar", "pnl", "nav")
    assert not any(any(b in k for b in banned) for k in out)
