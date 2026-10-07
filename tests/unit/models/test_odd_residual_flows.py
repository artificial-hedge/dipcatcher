"""Tests for quant_fund.models.odd_residual_flows — TORF (Madhusudhanan et al. 2026).

References: Madhusudhanan, Klötergens, Schmidt-Thieme & Yalavarthi (2026,
"Two-stage Odd Residual Flows for Mean-Preserving Probabilistic Time Series
Forecasting", arXiv:2608.11114); Kobayashi & Aotani (2023, Advanced Robotics
37:719-736, restricted/odd flows); Durkan, Bekasov, Murray & Papamakarios
(2019, NeurIPS 32, rational-quadratic splines); Stirn et al. (2023, AISTATS,
faithful heteroscedastic regression); Seitzer et al. (2022, ICLR, pitfalls of
joint NLL training); Gneiting & Raftery (2007, JASA 102:359-378, CRPS); Duan
et al. (2020, ICML, NGBoost baseline).

All data here is SYNTHETIC (seeded heteroskedastic streams, adapted from the
paper's Appendix C DGP) — algorithmic correctness evidence, never market
evidence; no live-trading claims. Proper scores only (CRPS, log score, PIT,
point MAE). Torch-dependent tests skip cleanly when the nn extra is absent;
the numpy core and all input-contract edges run without torch.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from typing import Any

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.metrics.probability import pit_ks
from quant_fund.metrics.scoring import crps_empirical, crps_gaussian
from quant_fund.models import odd_residual_flows as orf
from quant_fund.models.ngboost_lite import NGBoostGaussian
from quant_fund.models.odd_residual_flows import (
    OddResidualFlow,
    RidgeMeanForecaster,
    TORFForecaster,
)


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="TORF stage-2 training requires the nn extra (torch)"
)

# Documented tolerances.
# CRPS comparison slack: every run here is fully seeded and deterministic, so
# the slack only absorbs cross-platform float32 training jitter (~1e-3). The
# reference margins are 1.2e-2 (student_t) and 9.0e-2 (bimodal) — >= 6x the
# slack — so the paper's claim is asserted strictly, with the slack as guard.
CRPS_TOL = 2e-3
# Mean-preservation Monte-Carlo tolerance: 6x the standard error of the sample
# mean (false-alarm probability ~1e-9 per observation under the CLT).
MC_K = 6.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _gauss_resid(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Heteroskedastic GAUSSIAN residuals: the K=0 flow's exact model class."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 3))
    sig = 0.3 + 1.2 * (X[:, 1] + 1.0) / 2.0
    return X, sig * rng.standard_normal(n)


def _random_spline_params(
    rng: np.random.Generator, n: int, n_bins: int, bound: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = bound * orf._softmax_floor(rng.normal(size=(n, n_bins)), n_bins)
    h = bound * orf._softmax_floor(rng.normal(size=(n, n_bins)), n_bins)
    d_inner = np.logaddexp(0.0, rng.normal(size=(n, n_bins - 1))) + orf._DERIV_FLOOR
    d = np.concatenate([np.ones((n, 1)), d_inner, np.ones((n, 1))], axis=-1)
    return w, h, d


def _handmade_flow(seed: int = 0, n_blocks: int = 2, n_bins: int = 5) -> OddResidualFlow:
    """An OddResidualFlow with hand-injected numpy params — exercises the whole
    numpy inference path (forward/inverse/CDF/quantile/CRPS) WITHOUT torch."""
    rng = np.random.default_rng(seed)
    p = 2
    scale_layers = (
        (rng.normal(size=(8, p)) / math.sqrt(p), rng.normal(size=8)),
        (rng.normal(size=(1, 8)) / math.sqrt(8), np.zeros(1)),
    )
    spline_layers = tuple(
        (
            (rng.normal(size=(8, p)) / math.sqrt(p), rng.normal(size=8)),
            (
                rng.normal(size=(3 * n_bins - 1, 8)) / math.sqrt(8),
                rng.normal(size=3 * n_bins - 1) * 0.1,
            ),
        )
        for _ in range(n_blocks)
    )
    flow = OddResidualFlow(n_blocks=n_blocks, n_bins=n_bins, tail_bound=4.0)
    flow._params = orf._NumpyFlowParams(
        n_blocks=n_blocks,
        n_bins=n_bins,
        tail_bound=4.0,
        log_scale_bound=2.5,
        resid_std=1.3,
        ctx_mean=np.zeros(p),
        ctx_std=np.ones(p),
        scale_layers=scale_layers,
        spline_layers=spline_layers,
    )
    return flow


# ---------------------------------------------------------------------------
# numpy core (always run, no torch needed)
# ---------------------------------------------------------------------------


def test_rq_spline_uniform_params_are_identity_with_identity_tail() -> None:
    """Uniform bins + unit derivatives = the identity spline (Durkan et al. 2019),
    including the identity default outside [0, B]."""
    bound = 5.0
    n_bins = 8
    w = np.full(n_bins, bound / n_bins)
    h = np.full(n_bins, bound / n_bins)
    d = np.ones(n_bins + 1)
    x = np.array([0.0, 0.7, 2.5, 4.999, 5.0, 7.3])
    y, deriv = orf._rq_spline_forward_mag(x, w, h, d, bound=bound)
    np.testing.assert_allclose(y, x, atol=1e-12)
    np.testing.assert_allclose(deriv, 1.0, atol=1e-12)
    np.testing.assert_allclose(orf._rq_spline_inverse_mag(x, w, h, d, bound=bound), x, atol=1e-12)


def test_rq_spline_roundtrip_monotone_derivative_matches_fd() -> None:
    rng = np.random.default_rng(0)
    n_bins, bound = 6, 4.0
    w, h, d = _random_spline_params(rng, 5, n_bins, bound)
    x = np.sort(rng.uniform(0.0, bound, size=(5, 200)), axis=1)
    y, deriv = orf._rq_spline_forward_mag(x, w, h, d, bound=bound)
    assert np.all(np.diff(y, axis=1) > 0.0)  # strictly monotone
    assert np.all(deriv > 0.0)
    back = orf._rq_spline_inverse_mag(y, w, h, d, bound=bound)
    np.testing.assert_allclose(back, x, rtol=1e-9, atol=1e-9)
    # endpoints pinned: S(0) = 0, S(B) = B
    edge = np.array([[0.0, bound]]).repeat(5, axis=0)
    ye, _ = orf._rq_spline_forward_mag(edge, w, h, d, bound=bound)
    np.testing.assert_allclose(ye, edge, atol=1e-12)
    # analytic derivative vs central finite differences in the interior
    eps = 1e-6
    xc = np.clip(x, eps * 10, bound - eps * 10)
    fp = orf._rq_spline_forward_mag(xc + eps, w, h, d, bound=bound)[0]
    fm = orf._rq_spline_forward_mag(xc - eps, w, h, d, bound=bound)[0]
    _, dv = orf._rq_spline_forward_mag(xc, w, h, d, bound=bound)
    np.testing.assert_allclose((fp - fm) / (2 * eps), dv, rtol=1e-4)


def test_ross_odd_by_construction_sign_restore() -> None:
    """ROSS (paper Eqs. 7-8): v = sgn(u) S(|u|) is EXACTLY odd with an even
    log-derivative; sgn(0) = +1 keeps S(0) = 0 continuous."""
    rng = np.random.default_rng(1)
    n_bins, bound = 6, 4.0
    w, h, d = _random_spline_params(rng, 5, n_bins, bound)
    u = rng.normal(scale=2.0, size=(5, 400))
    v, ldj = orf._ross_forward(u, w, h, d, bound=bound)
    v_neg, ldj_neg = orf._ross_forward(-u, w, h, d, bound=bound)
    assert np.array_equal(v_neg, -v)  # exact oddness, no tolerance
    assert np.array_equal(ldj_neg, ldj)  # even log-Jacobian
    u0, ldj0 = orf._ross_forward(np.zeros((5, 1)), w, h, d, bound=bound)
    assert np.all(u0 == 0.0) and np.all(ldj0 == 0.0)  # fixed point at 0, S'(0)=1
    back = orf._ross_inverse(v, w, h, d, bound=bound)
    np.testing.assert_allclose(back, u, rtol=1e-9, atol=1e-9)
    # strictly increasing across the whole real line (monotone spline + sign restore)
    order = np.argsort(u, axis=1)
    v_sorted = np.take_along_axis(v, order, axis=1)
    assert np.all(np.diff(v_sorted, axis=1) > 0.0)


def test_handmade_flow_oddness_cdf_symmetry_median_and_quantile_duality() -> None:
    """The full numpy flow map (LSL+ROSS composition) is odd WITHOUT training:
    f(-r) = -f(r) exactly, F(-v) = 1 - F(v), Q(0.5) = 0 exactly, and
    F(Q(tau)) = tau (Lemma 1 machinery, arXiv:2608.11114 Appendix B)."""
    flow = _handmade_flow(seed=0)
    rng = np.random.default_rng(2)
    C = rng.normal(size=(7, 2))
    r = rng.normal(scale=1.5, size=7)
    z_pos, ldj_pos = flow.forward(C, r)
    z_neg, ldj_neg = flow.forward(C, -r)
    assert np.array_equal(z_neg, -z_pos)
    assert np.array_equal(ldj_neg, ldj_pos)
    np.testing.assert_allclose(flow.inverse(C, z_pos), r, rtol=1e-8, atol=1e-8)
    # CDF symmetry about zero (odd map + symmetric Gaussian base)
    grid = np.linspace(-5.0, 5.0, 41)
    C_rep = np.repeat(C[:1], grid.size, axis=0)
    F = flow.cdf(C_rep, grid)
    np.testing.assert_allclose(flow.cdf(C_rep, -grid), 1.0 - F, atol=1e-12)
    assert np.all((F >= 0.0) & (F <= 1.0))
    assert np.all(np.diff(F) >= 0.0)  # monotone (float underflow may flatten far tails)
    assert F[20] == 0.5  # grid[20] = 0 and F(0) = Phi(f(0)) = Phi(0) EXACTLY
    # zero median exact; quantile antisymmetry; CDF/quantile duality
    taus = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    q = flow.quantile(C[:2], taus)
    assert np.all(q[:, 3] == 0.0)
    np.testing.assert_allclose(q[:, :3], -q[:, ::-1][:, :3], atol=1e-9)
    for i in range(2):
        Fi = flow.cdf(np.repeat(C[i : i + 1], taus.size, axis=0), q[i])
        np.testing.assert_allclose(Fi, taus, atol=1e-8)
    # log density equals dF/dv (finite differences)
    v = rng.uniform(-2.0, 2.0, size=7)
    p_fd = (flow.cdf(C, v + 1e-6) - flow.cdf(C, v - 1e-6)) / 2e-6
    np.testing.assert_allclose(np.exp(flow.log_prob(C, v)), p_fd, rtol=1e-4)


def test_handmade_flow_crps_positive_quadrature_converged_and_minimized_at_median() -> None:
    """CRPS via the documented CDF-integral Gauss-Legendre quadrature:
    positive, converged under refinement, and (proper score) minimized at the
    predictive median, which oddness pins to zero."""
    flow = _handmade_flow(seed=1)
    C = np.zeros((3, 2))  # identical contexts -> identical predictive law
    r = np.array([0.0, 1.0, 4.0])
    cr = flow.crps(C, r)
    assert np.all(np.isfinite(cr)) and np.all(cr > 0.0)
    assert cr[0] < cr[1] < cr[2]  # minimized at the (zero) median, grows in the tail
    refined = flow.crps(C, r, n_panels=160, n_nodes=16)
    np.testing.assert_allclose(cr, refined, rtol=1e-5, atol=1e-7)
    with pytest.raises(ValueError, match="n_panels"):
        flow.crps(C, r, n_panels=0)
    with pytest.raises(ValueError, match="tail_tau"):
        flow.crps(C, r, tail_tau=0.5)


# ---------------------------------------------------------------------------
# stage 1: ridge mean baseline (numpy, deterministic)
# ---------------------------------------------------------------------------


def test_ridge_recovers_linear_mean_and_is_bitwise_deterministic() -> None:
    rng = np.random.default_rng(4)
    beta = np.array([1.5, -2.0, 0.5])
    X = rng.uniform(-1.0, 1.0, size=(400, 3))
    y = X @ beta + 0.7 + 0.05 * rng.standard_normal(400)
    m = RidgeMeanForecaster(alpha=1e-3).fit(X, y)
    pred = m.predict(X)
    assert np.corrcoef(pred, y)[0, 1] > 0.999
    assert float(np.mean(np.abs(y - pred))) < 0.06
    again = RidgeMeanForecaster(alpha=1e-3).fit(X, y)
    assert np.array_equal(again.predict(X), pred)  # closed form: no RNG at all
    Xt = rng.uniform(-1.0, 1.0, size=(100, 3))
    yt = Xt @ beta + 0.7 + 0.05 * rng.standard_normal(100)
    assert float(np.mean(np.abs(yt - m.predict(Xt)))) < 0.08


def test_ridge_fail_closed_edges() -> None:
    rng = np.random.default_rng(5)
    X = rng.normal(size=(20, 2))
    y = rng.normal(size=20)
    with pytest.raises(ValueError, match="alpha"):
        RidgeMeanForecaster(alpha=-1.0)
    with pytest.raises(ValueError, match="alpha"):
        RidgeMeanForecaster(alpha=np.nan)
    m = RidgeMeanForecaster()
    with pytest.raises(RuntimeError, match="not fitted"):
        m.predict(X)
    with pytest.raises(ValueError, match="2-D"):
        m.fit(X.ravel(), y)
    with pytest.raises(ValueError, match="length"):
        m.fit(X, y[:-1])
    with pytest.raises(ValueError, match="finite"):
        m.fit(X, np.r_[np.inf, y[1:]])
    with pytest.raises(ValueError, match="too few"):
        m.fit(X[:3], y[:3])
    # a zero-variance column makes the normal equations EXACTLY singular at
    # alpha = 0 (exact zeros survive any BLAS/FMA association; a merely
    # collinear column does not)
    Xz = np.column_stack([X[:, 0], np.zeros(20)])
    with pytest.raises(ValueError, match="singular"):
        RidgeMeanForecaster(alpha=0.0).fit(Xz, y)
    m.fit(X, y)
    with pytest.raises(ValueError, match="feature count"):
        m.predict(X[:, :1])


# ---------------------------------------------------------------------------
# fail-closed constructor/contract edges (torch-free by design)
# ---------------------------------------------------------------------------


def test_flow_constructor_fail_closed() -> None:
    for kw in (
        {"n_blocks": -1},
        {"n_blocks": True},
        {"n_bins": 0},
        {"tail_bound": 0.0},
        {"tail_bound": float("inf")},
        {"hidden": ()},
        {"hidden": (0,)},
        {"hidden": (4, -2)},
        {"log_scale_bound": 0.0},
        {"epochs": 0},
        {"lr": 0.0},
        {"lr": -1.0},
    ):
        with pytest.raises(ValueError):
            OddResidualFlow(**kw)  # type: ignore[arg-type]


def test_flow_fit_input_validation_runs_without_torch() -> None:
    """Input-contract errors raise ValueError BEFORE torch is touched, so the
    fail-closed surface is testable (and holds) even without the nn extra."""
    flow = OddResidualFlow(epochs=2)
    rng = np.random.default_rng(6)
    C = rng.normal(size=(32, 2))
    r = rng.normal(size=32)
    with pytest.raises(RuntimeError, match="not fitted"):
        flow.cdf(C, r)
    with pytest.raises(ValueError, match="2-D"):
        flow.fit(C.ravel(), r)
    with pytest.raises(ValueError, match="length"):
        flow.fit(C, r[:-1])
    with pytest.raises(ValueError, match="finite"):
        flow.fit(C, np.r_[np.nan, r[1:]])
    with pytest.raises(ValueError, match="too few"):
        flow.fit(C[:4], r[:4])
    with pytest.raises(ValueError, match="zero variance"):
        flow.fit(C, np.zeros(32))


def test_torf_init_and_unfitted_fail_closed() -> None:
    with pytest.raises(ValueError, match="fit"):
        TORFForecaster(stage1=object())
    with pytest.raises(ValueError, match="OddResidualFlow"):
        TORFForecaster(flow="not a flow")  # type: ignore[arg-type]
    t = TORFForecaster()
    X = np.zeros((6, 2))
    with pytest.raises(RuntimeError, match="not fitted"):
        t.predict(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        t.crps(X, np.zeros(6))
    with pytest.raises(RuntimeError, match="not fitted"):
        t.sample(X, 10)
    # zero-variance residuals fail closed inside fit, before torch
    with pytest.raises(ValueError, match="zero variance"):
        TORFForecaster().fit(np.zeros((20, 1)), np.full(20, 3.0))


def test_module_imports_without_torch_and_raises_clear_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No top-level torch import; the numpy core stays usable with torch
    blocked; every training entry point fails closed with nn-extra guidance."""
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    spec = importlib.util.spec_from_file_location("_orf_no_torch", orf.__file__)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_orf_no_torch", probe)
    spec.loader.exec_module(probe)  # must import cleanly without torch
    # numpy core works while torch is blocked
    bound, n_bins = 3.0, 4
    w = np.full(n_bins, bound / n_bins)
    y, deriv = probe._rq_spline_forward_mag(
        np.array([0.5, 1.5, 2.9, 4.0]), w, w.copy(), np.ones(n_bins + 1), bound=bound
    )
    np.testing.assert_allclose(y, np.array([0.5, 1.5, 2.9, 4.0]), atol=1e-12)
    np.testing.assert_allclose(deriv, 1.0, atol=1e-12)
    rng = np.random.default_rng(7)
    C, r = rng.normal(size=(16, 2)), rng.normal(size=16)
    with pytest.raises(ImportError, match=r"'nn' extra"):
        probe.OddResidualFlow(epochs=2).fit(C, r)
    with pytest.raises(ImportError, match=r"'nn' extra"):
        probe.TORFForecaster().fit(C, r)
    with pytest.raises(ImportError, match=r"'nn' extra"):
        probe.bench_odd_residual_flows(n_train=10, n_test=6, epochs=2, ngboost_rounds=2)


# ---------------------------------------------------------------------------
# torch lane (skipped cleanly when the nn extra is absent)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_student_t() -> dict[str, Any]:
    """TORF (K=2), its own K=0 Gaussian ablation (MVE-2S, Appendix G), and a
    CRPS-trained NGBoostGaussian on the same seeded heavy-tailed stream."""
    if not _HAS_TORCH:
        pytest.skip("TORF stage-2 training requires the nn extra (torch)")
    Xtr, ytr, Xte, yte = orf._synthetic_hetero_stream(1000, 600, seed=0, noise="student_t")
    torf = TORFForecaster(flow=OddResidualFlow(n_blocks=2, n_bins=8, epochs=250, seed=0)).fit(
        Xtr, ytr
    )
    torf_k0 = TORFForecaster(flow=OddResidualFlow(n_blocks=0, epochs=250, seed=0)).fit(Xtr, ytr)
    ngb = NGBoostGaussian(n_estimators=80, learning_rate=0.1, score="crps", seed=0).fit(Xtr, ytr)
    return {
        "Xtr": Xtr,
        "ytr": ytr,
        "Xte": Xte,
        "yte": yte,
        "torf": torf,
        "torf_k0": torf_k0,
        "ngb": ngb,
    }


@pytest.fixture(scope="module")
def fitted_bimodal() -> dict[str, Any]:
    """Same comparison on the symmetric bimodal regime, where a Gaussian
    residual head is structurally mis-shaped."""
    if not _HAS_TORCH:
        pytest.skip("TORF stage-2 training requires the nn extra (torch)")
    Xtr, ytr, Xte, yte = orf._synthetic_hetero_stream(1000, 600, seed=0, noise="bimodal")
    torf = TORFForecaster(flow=OddResidualFlow(n_blocks=2, n_bins=8, epochs=250, seed=0)).fit(
        Xtr, ytr
    )
    torf_k0 = TORFForecaster(flow=OddResidualFlow(n_blocks=0, epochs=250, seed=0)).fit(Xtr, ytr)
    ngb = NGBoostGaussian(n_estimators=80, learning_rate=0.1, score="crps", seed=0).fit(Xtr, ytr)
    return {"Xte": Xte, "yte": yte, "torf": torf, "torf_k0": torf_k0, "ngb": ngb}


@requires_torch
def test_k0_reduces_to_conditional_gaussian_appendix_g() -> None:
    """Appendix G: with K=0 the flow IS a zero-mean conditional Gaussian
    (MVE-2S), so every numpy inference channel must agree with the closed-form
    Gaussian to quadrature precision, and training must start exactly at the
    Gaussian initialization (identity splines, unit scales)."""
    X, r = _gauss_resid(800, 3)
    flow = OddResidualFlow(n_blocks=0, hidden=(16, 16), epochs=300, lr=1e-2, seed=0).fit(X, r)
    info = flow.fit_info
    assert info is not None
    # identity-at-init: loss_curve[0] is the standardized N(0,1) NLL exactly
    assert info.loss_curve[0] == pytest.approx(0.5 + 0.5 * math.log(2.0 * math.pi), abs=0.01)
    assert info.loss_curve[-1] < info.loss_curve[0]
    # torch->numpy extraction parity: raw NLL = standardized NLL + log(resid_std)
    assert abs(info.final_loss - math.log(flow.resid_std) - info.loss_curve[-1]) < 5e-3
    n = r.size
    sig_hat = flow.quantile(X, np.array([0.9]))[:, 0] / norm.ppf(0.9)
    assert np.all(sig_hat > 0.0)
    assert np.corrcoef(sig_hat, 0.3 + 1.2 * (X[:, 1] + 1.0) / 2.0)[0, 1] > 0.85  # tracks sigma(x)
    taus = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    np.testing.assert_allclose(
        flow.quantile(X, taus), sig_hat[:, None] * norm.ppf(taus)[None, :], atol=1e-9
    )
    np.testing.assert_allclose(flow.cdf(X, r), norm.cdf(r / sig_hat), atol=1e-12)
    np.testing.assert_allclose(flow.crps(X, r), crps_gaussian(r, np.zeros(n), sig_hat), atol=1e-11)
    np.testing.assert_allclose(
        flow.log_prob(X, r),
        -0.5 * np.log(2.0 * math.pi) - np.log(sig_hat) - 0.5 * (r / sig_hat) ** 2,
        atol=1e-11,
    )


@requires_torch
def test_training_determinism_and_seed_sensitivity() -> None:
    X, r = _gauss_resid(400, 11)
    kw = dict(n_blocks=1, n_bins=4, hidden=(16, 16), epochs=100, lr=8e-3)
    a = OddResidualFlow(seed=5, **kw).fit(X, r)
    b = OddResidualFlow(seed=5, **kw).fit(X, r)
    c = OddResidualFlow(seed=6, **kw).fit(X, r)
    taus = np.array([0.1, 0.5, 0.9])
    assert np.array_equal(a.quantile(X[:32], taus), b.quantile(X[:32], taus))
    assert np.array_equal(a.crps(X[:32], r[:32]), b.crps(X[:32], r[:32]))
    assert a.fit_info is not None and b.fit_info is not None
    assert a.fit_info.loss_curve == b.fit_info.loss_curve
    assert not np.array_equal(a.quantile(X[:32], taus), c.quantile(X[:32], taus))


@requires_torch
def test_fitted_flow_structural_guarantees(fitted_student_t: dict[str, Any]) -> None:
    """Oddness, exact zero median, and inverse-pass roundtrip on FITTED params."""
    torf: TORFForecaster = fitted_student_t["torf"]
    flow = torf.flow
    C = fitted_student_t["Xte"][:24]
    rng = np.random.default_rng(9)
    r = rng.normal(scale=1.2, size=24)
    z_pos, ldj_pos = flow.forward(C, r)
    z_neg, ldj_neg = flow.forward(C, -r)
    assert np.array_equal(z_neg, -z_pos)  # exact by construction, not tolerance
    assert np.array_equal(ldj_neg, ldj_pos)
    np.testing.assert_allclose(flow.inverse(C, z_pos), r, rtol=1e-6, atol=1e-9)
    # predictive median equals the point forecast EXACTLY (Lemma 1: zero median)
    mu = torf.predict(C)
    q_med = torf.predict_quantiles(C, np.array([0.5]))[:, 0]
    assert np.array_equal(q_med, mu)
    # predictive quantiles never cross
    q = torf.predict_quantiles(C, np.array([0.05, 0.25, 0.5, 0.75, 0.95]))
    assert np.all(np.diff(q, axis=1) > 0.0)


@requires_torch
def test_mean_preservation_on_samples_documented_mc_tolerance(
    fitted_student_t: dict[str, Any],
) -> None:
    """E[eps | x] = 0 under the learned odd density (arXiv:2608.11114 Lemma 1),
    asserted on seeded inverse-pass samples.

    Tolerance: MC_K=6 times the standard error of the sample mean per
    observation (false-alarm ~1e-9 under the CLT); the reference run passed
    with max ratio 0.28 at M=20000 even in the heavy-tailed regime. Oddness
    makes this exact in the population — the bound only charges sampling
    noise, never model bias.
    """
    torf: TORFForecaster = fitted_student_t["torf"]
    C = fitted_student_t["Xte"][:40]
    m = 20000
    eps = torf.flow.sample(C, m, seed=7)
    assert eps.shape == (40, m)
    per_se = eps.std(axis=1) / math.sqrt(m)
    assert np.all(np.abs(eps.mean(axis=1)) <= MC_K * per_se)
    pooled = float(np.mean(eps))
    pooled_se = float(np.std(eps) / math.sqrt(eps.size))
    assert abs(pooled) <= MC_K * pooled_se
    # predictive mean = stage-1 forecast under sampling too (translation by mu)
    mu = torf.predict(C)
    draws = torf.sample(C, 4000, seed=8)
    dev = draws.mean(axis=1) - mu
    assert np.all(np.abs(dev) <= MC_K * draws.std(axis=1) / math.sqrt(4000))
    # seeded sampler determinism
    assert np.array_equal(torf.flow.sample(C, 64, seed=3), torf.flow.sample(C, 64, seed=3))
    assert not np.array_equal(torf.flow.sample(C, 64, seed=3), torf.flow.sample(C, 64, seed=4))


@requires_torch
def test_crps_quadrature_matches_sampling_within_mc_tolerance(
    fitted_student_t: dict[str, Any],
) -> None:
    """Deterministic CDF quadrature vs the sampling-based estimator the paper
    uses (M=100 empirical CDF, Appendix F). Cross-checked against the repo's
    O(M^2) ``crps_empirical`` at M=1500; tolerance 0.02 is ~5x the |diff| =
    0.004 of the seeded reference run (MC noise of the empirical mean)."""
    torf: TORFForecaster = fitted_student_t["torf"]
    sub = fitted_student_t["Xte"][:120]
    r = fitted_student_t["yte"][:120] - torf.predict(sub)
    quad = torf.flow.crps(sub, r)
    samp = torf.flow.sample(sub, 1500, seed=5)
    emp = np.array([crps_empirical(r[i], samp[i]) for i in range(r.size)])
    assert abs(float(quad.mean()) - float(emp.mean())) < 0.02
    # quadrature is converged: 4x refinement moves it by < 1e-4 per-observation
    # (reference worst-obs diff 1.4e-6) and < 1e-7 on the mean
    refined = torf.flow.crps(sub, r, n_panels=160, n_nodes=24)
    np.testing.assert_allclose(quad, refined, rtol=1e-4, atol=5e-6)
    assert abs(quad.mean() - refined.mean()) <= 1e-7 * refined.mean()


@requires_torch
def test_torf_beats_ngboost_and_preserves_stage1_exactly_student_t(
    fitted_student_t: dict[str, Any],
) -> None:
    """The paper's claim on a seeded SYNTHETIC heavy-tailed heteroskedastic
    stream (Appendix C-style): TORF CRPS <= NGBoostGaussian CRPS under the
    SAME proper score, with the Stage-1 point forecast preserved EXACTLY
    (equality, not tolerance — the mean is never re-derived from the density).

    Reference margins (seed 0): NGBoost +1.2e-2 (2.3%), MVE-2S K=0 +1e-4;
    the CRPS_TOL=2e-3 slack only absorbs cross-platform float32 jitter.
    """
    torf: TORFForecaster = fitted_student_t["torf"]
    torf_k0: TORFForecaster = fitted_student_t["torf_k0"]
    ngb: NGBoostGaussian = fitted_student_t["ngb"]
    Xte, yte = fitted_student_t["Xte"], fitted_student_t["yte"]
    torf_crps = torf.crps(Xte, yte)
    ngb_crps = ngb.crps(Xte, yte)
    assert torf_crps < ngb_crps
    assert torf_crps <= ngb_crps + CRPS_TOL
    assert torf_crps <= torf_k0.crps(Xte, yte) + CRPS_TOL  # flexible >= Gaussian 2-stage
    # EXACT mean preservation: TORF's point forecast IS the frozen Stage-1 output
    mu_torf = torf.predict(Xte)
    assert np.array_equal(mu_torf, np.asarray(torf.stage1.predict(Xte), dtype=float))
    assert torf.mae(Xte, yte) == float(np.mean(np.abs(yte - mu_torf)))
    # calibration + density quality (proper scores only)
    q = torf.predict_quantiles(Xte, np.array([0.05, 0.95]))
    cov = float(np.mean((yte >= q[:, 0]) & (yte <= q[:, 1])))
    assert 0.83 <= cov <= 0.95
    _, p = pit_ks(torf.pit(Xte, yte))
    assert p > 0.01
    assert torf.log_score(Xte, yte) >= ngb.log_score(Xte, yte) - 5e-3


@requires_torch
def test_torf_wins_big_on_symmetric_bimodal_regime(fitted_bimodal: dict[str, Any]) -> None:
    """Symmetric bimodal residuals: a Gaussian head (NGBoost or the K=0
    ablation) is structurally mis-shaped, while the odd spline flow captures
    both modes and still preserves the mean exactly. Reference margins
    (seed 0): NGBoost +9.0e-2, MVE-2S +5.8e-2; asserted at >= 2e-2 (3x+)."""
    torf: TORFForecaster = fitted_bimodal["torf"]
    torf_k0: TORFForecaster = fitted_bimodal["torf_k0"]
    ngb: NGBoostGaussian = fitted_bimodal["ngb"]
    Xte, yte = fitted_bimodal["Xte"], fitted_bimodal["yte"]
    torf_crps = torf.crps(Xte, yte)
    assert torf_crps <= ngb.crps(Xte, yte) - 2e-2
    assert torf_crps <= torf_k0.crps(Xte, yte) - 2e-2
    mu = torf.predict(Xte)
    assert np.array_equal(mu, np.asarray(torf.stage1.predict(Xte), dtype=float))
    assert torf.mae(Xte, yte) == float(np.mean(np.abs(yte - mu)))
    q = torf.predict_quantiles(Xte, np.array([0.05, 0.5, 0.95]))
    assert np.all(np.diff(q, axis=1) > 0.0)
    np.testing.assert_allclose(q[:, 1], mu, atol=0.0, rtol=0.0)


@requires_torch
def test_fitted_fail_closed_edges(fitted_student_t: dict[str, Any]) -> None:
    torf: TORFForecaster = fitted_student_t["torf"]
    flow = torf.flow
    X = fitted_student_t["Xte"]
    n = X.shape[0]
    with pytest.raises(ValueError, match="feature count"):
        flow.cdf(X[:, :2], np.zeros(n))
    with pytest.raises(ValueError, match="feature count"):
        torf.predict(X[:, :2])
    with pytest.raises(ValueError, match="2-D"):
        torf.predict(X.ravel())
    with pytest.raises(ValueError, match="length"):
        flow.forward(X, np.zeros(n - 1))
    with pytest.raises(ValueError, match="finite"):
        flow.cdf(X, np.full(n, np.nan))
    with pytest.raises(ValueError, match="taus"):
        flow.quantile(X, np.array([0.0, 0.5]))
    with pytest.raises(ValueError, match="taus"):
        flow.quantile(X, np.array([1.0]))
    with pytest.raises(ValueError, match="taus"):
        flow.quantile(X, np.array([np.nan]))
    with pytest.raises(ValueError, match="taus"):
        flow.quantile(X, np.array([]))
    with pytest.raises(ValueError, match="n_samples"):
        flow.sample(X, 0)


@requires_torch
def test_bench_keys_labels_and_claims() -> None:
    """bench_odd_residual_flows: scorecard-ready proper-score keys, honesty
    labels, exact MAE preservation, and TORF >= NGBoost on CRPS. SYNTHETIC
    fixture DGP — never market evidence."""
    out = orf.bench_odd_residual_flows(n_train=1000, n_test=600, seed=0, epochs=200)
    expected = {
        "synthetic_torf_crps",
        "synthetic_ngboost_crps",
        "synthetic_crps_gain_vs_ngboost",
        "synthetic_torf_log_score",
        "synthetic_ngboost_log_score",
        "synthetic_torf_mae",
        "synthetic_stage1_mae",
        "synthetic_mae_preservation_gap",
        "synthetic_residual_sample_mean",
        "synthetic_coverage_90",
        "synthetic_pit_ks",
        "synthetic_pit_ks_pvalue",
        "synthetic_n_train",
        "synthetic_n_test",
        "synthetic_seed",
        "synthetic_noise",
        "synthetic_dgp",
        "synthetic_claim",
        "synthetic_synthetic",
    }
    assert set(out) == expected
    assert out["synthetic_dgp"] == "fixture"
    assert out["synthetic_claim"] == "research_metric_only"
    assert out["synthetic_synthetic"] == "heteroskedastic_symmetric_seeded"
    assert out["synthetic_mae_preservation_gap"] == 0.0
    assert out["synthetic_torf_mae"] == out["synthetic_stage1_mae"]
    assert out["synthetic_crps_gain_vs_ngboost"] > 0.0
    assert out["synthetic_torf_crps"] <= out["synthetic_ngboost_crps"] + CRPS_TOL
    assert 0.80 <= out["synthetic_coverage_90"] <= 0.97
    assert abs(out["synthetic_residual_sample_mean"]) <= 0.02
    assert out["synthetic_pit_ks_pvalue"] > 0.01
    for key, value in out.items():
        if isinstance(value, float):
            assert math.isfinite(value), key
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav", "drawdown")
    assert not any(tok in key.lower() for key in out for tok in forbidden)


@requires_torch
def test_injected_stage1_is_preserved_verbatim() -> None:
    """TORF is Stage-1 model-agnostic (paper §4): any fit/predict point
    forecaster is kept EXACTLY as the predictive mean by the odd flow."""

    class ConstPlusFeature:
        def __init__(self) -> None:
            self.c = 0.0

        def fit(self, X: np.ndarray, y: np.ndarray) -> ConstPlusFeature:
            self.c = float(np.mean(y) - np.mean(X[:, 0]))
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.asarray(self.c + np.asarray(X, dtype=float)[:, 0], dtype=float)

    Xtr, ytr, Xte, yte = orf._synthetic_hetero_stream(300, 200, seed=2, noise="student_t")
    stage1 = ConstPlusFeature()
    torf = TORFForecaster(
        stage1=stage1, flow=OddResidualFlow(n_blocks=1, n_bins=4, hidden=(16,), epochs=80, seed=0)
    ).fit(Xtr, ytr)
    assert np.array_equal(torf.predict(Xte), stage1.predict(Xte))
    assert torf.mae(Xte, yte) == float(np.mean(np.abs(yte - stage1.predict(Xte))))
    q = torf.predict_quantiles(Xte, np.array([0.5]))[:, 0]
    assert np.array_equal(q, stage1.predict(Xte))  # median preserved too
