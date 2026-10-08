"""Tests for quant_fund.models.diffusion_forecaster — DiffPTS (Ye et al. 2026).

References: Ye, Li, Liu, Jiang, Sekimoto & Jiang (2026, "DiffPTS: Rethinking
Diffusion ELBO for Probabilistic Time Series Forecasting", arXiv:2609.32363,
NeurIPS 2026 Poster — Proposition 3.1 exact ELBO / Eq. 12, Proposition 3.3
joint objective / Eq. 14, Algorithm 2 sampler, Appendix A derivations,
Appendix E.4 weighting discussion); Ho, Jain & Abbeel (2020, NeurIPS 33,
DDPM); Nichol & Dhariwal (2021, ICML, cosine schedule); Gneiting & Raftery
(2007, JASA 102:359-378, CRPS); Duan et al. (2020, ICML, NGBoost baseline);
Madhusudhanan et al. (2026, arXiv:2608.11114, TORF baseline).

All data here is SYNTHETIC (seeded heteroskedastic streams composed in the
TORF/DeRegiME lane fixture style — bitwise-identical to
``odd_residual_flows._synthetic_hetero_stream`` for shared seeds, asserted
below) — algorithmic correctness evidence, never market evidence; no
live-trading claims. Proper scores only (CRPS, endpoint log score, PIT,
coverage) plus point MAE. Torch-dependent tests skip cleanly when the nn
extra is absent; the numpy core (schedules, Algorithm-2 sampler, scoring
wiring) and all input-contract edges run without torch.

Documented tolerances. Every run is fully seeded and deterministic, so the
CRPS slack only absorbs cross-platform float32 training jitter plus the
O(1/sqrt(M)) Monte-Carlo noise of the M-sample empirical CRPS estimator.
Reference margins at M=200 (seed 0): NGBoost +2.8e-2 (gauss), +1.3e-2
(student_t), +8.0e-2 (bimodal) — at least 4x the CRPS_TOL slack. The PIT-KS
statistic has an MC floor of ~0.032 at n=600 even under an EXACTLY calibrated
predictive (measured, M=400), so KS assertions carry that floor as headroom.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from typing import Any

import numpy as np
import pytest

from quant_fund.metrics.probability import pit_ks
from quant_fund.metrics.scoring import crps_empirical, crps_gaussian
from quant_fund.models import diffusion_forecaster as dfm
from quant_fund.models.diffusion_forecaster import (
    DiffPTSForecaster,
    bench_diffpts,
    noise_schedule,
)
from quant_fund.models.ngboost_lite import NGBoostGaussian


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="DiffPTS training requires the nn extra (torch)"
)

# Slack for seeded comparisons: cross-platform float32 training jitter plus
# M-sample empirical-CRPS Monte-Carlo noise (reference margins are >= 4x).
CRPS_TOL = 3e-3
# PIT-KS MC floor under an exactly calibrated predictive at n=600, M=400.
PIT_KS_FLOOR = 0.045


# ---------------------------------------------------------------------------
# helpers (SYNTHETIC fixtures, composed in the lane fixture style)
# ---------------------------------------------------------------------------


def _gauss_hetero_stream(
    n_train: int, n_test: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SYNTHETIC seeded heteroskedastic GAUSSIAN stream — the LSNM model class
    contains the truth here (``y = mu(x) + gamma(x) eps``), so calibration is
    testable at nominal levels. Same feature/mean/scale functions as the shared
    TORF-lane streams; only the noise law differs (correctness, never market
    evidence). Returns ``gamma`` on the test rows for sigma-tracking checks."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    x0 = rng.uniform(-1.0, 1.0, size=n)
    x1 = rng.uniform(-1.0, 1.0, size=n)
    mu = 2.0 * x0 + x1 * x1
    feats = np.column_stack([x0, x1, x1 * x1])
    gamma = 0.3 + 1.2 * (x1 + 1.0) / 2.0
    y = mu + gamma * rng.standard_normal(n)
    return (
        np.asarray(feats[:n_train], dtype=float),
        np.asarray(y[:n_train], dtype=float),
        np.asarray(feats[n_train:], dtype=float),
        np.asarray(y[n_train:], dtype=float),
        np.asarray(gamma[n_train:], dtype=float),
    )


def _handmade(
    seed: int = 0, n_steps: int = 3, p: int = 2, schedule: str = "cosine"
) -> DiffPTSForecaster:
    """A DiffPTSForecaster with hand-injected numpy params — exercises the
    whole numpy inference path (backbone, Algorithm-2 sampler, quantiles,
    PIT, CRPS wiring) WITHOUT torch."""
    rng = np.random.default_rng(seed)
    if schedule == "linear" and n_steps == 1:
        sched = noise_schedule(1, "linear", 0.3, 0.5)  # beta_1 = 0.3
    else:
        sched = noise_schedule(n_steps, schedule)
    h = 6
    f_layers = (
        (rng.normal(size=(h, p)) / math.sqrt(p), rng.normal(size=h) * 0.3),
        (rng.normal(size=(1, h)) / math.sqrt(h), np.zeros(1)),
    )
    g_layers = (
        (rng.normal(size=(h, p)) / math.sqrt(p), rng.normal(size=h) * 0.3),
        (rng.normal(size=(1, h)) / math.sqrt(h), np.zeros(1)),
    )
    d_in = 1 + n_steps + p
    # Single linear layer (no hidden) so the T=1 closed form stays analytic.
    eps_layers = ((rng.normal(size=(1, d_in)) / math.sqrt(d_in), np.zeros(1)),)
    model = DiffPTSForecaster(n_steps=n_steps, schedule=schedule)
    model._params = dfm._NumpyDiffPTSParams(
        schedule=sched,
        ablation="full",
        log_g_clip=12.0,
        y_mean=0.5,
        y_std=2.0,
        ctx_mean=np.zeros(p),
        ctx_std=np.ones(p),
        f_layers=f_layers,
        g_layers=g_layers,
        eps_layers=eps_layers,
        y0_clip_lo=-50.0,
        y0_clip_hi=50.0,
    )
    return model


# ---------------------------------------------------------------------------
# numpy core (always run, no torch needed)
# ---------------------------------------------------------------------------


def test_noise_schedule_invariants_and_paper_linear_abar_t() -> None:
    """Schedule algebra (paper Eqs. 9-12) and citation anchors: the paper's
    linear schedule (beta 1e-4 -> 0.02, T=20) gives abar_T ~ 0.82 exactly as
    quoted in arXiv:2609.32363 Appendix E.4; the cosine schedule (Nichol &
    Dhariwal 2021) clips betas at 0.999 and drives abar_T ~ 1e-5, so the
    learned endpoint N(f, g) nearly matches q(Y_T | Y_0, X)."""
    lin = noise_schedule(20, "linear", 1e-4, 0.02)
    cos = noise_schedule(20, "cosine")
    assert abs(lin.alpha_bar_T - 0.8168) < 5e-3  # paper App E.4: "~0.82"
    assert cos.alpha_bar_T < 1e-4
    assert float(cos.beta.max()) <= 0.999 + 1e-12
    for s in (lin, cos):
        assert np.all(s.beta > 0.0) and np.all(s.beta < 1.0)
        np.testing.assert_allclose(s.alpha, 1.0 - s.beta, atol=1e-15)
        np.testing.assert_allclose(s.alpha_bar, np.cumprod(s.alpha), atol=1e-15)
        assert np.all(np.diff(s.alpha_bar) < 0.0)  # strictly decreasing
        np.testing.assert_allclose(s.sqrt_beta_bar**2 + s.alpha_bar, 1.0, atol=1e-12)
        # beta_tilde_t = (1 - abar_{t-1}) / (1 - abar_t) * beta_t, zero at t=1
        assert s.beta_tilde[0] == 0.0
        ratio = (1.0 - s.alpha_bar[:-1]) / (1.0 - s.alpha_bar[1:])
        np.testing.assert_allclose(s.beta_tilde[1:], ratio * s.beta[1:], atol=1e-15)
        assert np.all(s.beta_tilde[1:] < s.beta[1:])
        # gamma_t exact-ELBO weights (Prop 3.1): gamma_1 = 1/(2 alpha_1)
        assert np.all(np.isfinite(s.gamma)) and np.all(s.gamma > 0.0)
        np.testing.assert_allclose(s.gamma[0], 1.0 / (2.0 * s.alpha[0]), atol=1e-15)
        prev = np.concatenate([[1.0], s.alpha_bar[:-1]])
        np.testing.assert_allclose(
            s.gamma[1:], s.beta[1:] / (2.0 * s.alpha[1:] * (1.0 - prev[1:])), atol=1e-12
        )
        # posterior-mean coefficients (Eq. 11): convex fixed point, and the
        # t=1 branch reduces to Y0_hat exactly (c0, c1, c2) = (1, 0, 0).
        np.testing.assert_allclose(s.c0 + s.c1 + s.c2, 1.0, atol=1e-12)
        assert s.c0[0] == pytest.approx(1.0, abs=1e-12)
        assert s.c1[0] == pytest.approx(0.0, abs=1e-12)
        assert s.c2[0] == pytest.approx(0.0, abs=1e-12)
        assert float(np.min(np.concatenate([s.c0, s.c1, s.c2]))) > -1.0
    with pytest.raises(ValueError, match="n_steps"):
        noise_schedule(0)
    with pytest.raises(ValueError, match="schedule"):
        noise_schedule(5, "vsde")
    with pytest.raises(ValueError, match="beta_start"):
        noise_schedule(5, "linear", 0.5, 0.1)
    with pytest.raises(ValueError, match="beta_end"):
        noise_schedule(5, "linear", 0.5, 1.5)


def test_posterior_mean_fixed_point() -> None:
    """mu_tilde_t(c, c, X) with f = c returns c exactly (c0+c1+c2 = 1): the
    reverse posterior mean of Eq. 11 preserves constant trajectories, the
    algebraic consistency condition the sampler relies on."""
    s = noise_schedule(10, "cosine")
    c = 1.7
    mu = s.c0 * c + s.c1 * c + s.c2 * c
    np.testing.assert_allclose(mu, c, atol=1e-12)


def test_require_finite_loss_guard_fails_closed() -> None:
    """The training-loop NaN/inf guard (fail-closed on non-finite loss)."""
    assert dfm._require_finite_loss(1.25, 0) == 1.25
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="not finite"):
            dfm._require_finite_loss(bad, 7)


def test_stream_is_bitwise_identical_to_torf_lane_fixture() -> None:
    """The shared SYNTHETIC streams: same DGP, same rng call order, same
    seeds => bitwise-identical data to the TORF lane's fixture, so the
    DiffPTS/TORF/NGBoost comparison runs on the SAME streams."""
    from quant_fund.models import odd_residual_flows as orf

    for noise in ("student_t", "bimodal"):
        mine = dfm._synthetic_hetero_stream(50, 30, 1, noise)
        theirs = orf._synthetic_hetero_stream(50, 30, 1, noise)
        assert all(np.array_equal(a, b) for a, b in zip(mine, theirs, strict=True))
    with pytest.raises(ValueError, match="noise"):
        dfm._synthetic_hetero_stream(10, 5, 0, "laplace")


def test_handmade_single_step_sampling_matches_closed_form() -> None:
    """Algorithm 2 at T=1 with a linear eps net: the seeded sampler must
    reproduce the closed-form composition (prior draw -> eps prediction ->
    x0-hat inversion of Eq. 20/21 -> t=1 branch) EXACTLY, with no reverse
    noise drawn (the t=1 branch returns Y0_hat)."""
    model = _handmade(seed=0, n_steps=1, schedule="linear")
    params = model._params
    assert params is not None
    rng = np.random.default_rng(11)
    X = rng.normal(size=(5, 2))
    samples = model.predict_samples(X, 7, seed=3)
    assert samples.shape == (5, 7)
    # manual re-derivation (no module helpers beyond the schedule arrays)
    Xs = X  # ctx_mean=0, ctx_std=1
    (W1, b1), (W2, b2) = params.f_layers  # type: ignore[misc]
    f = (np.tanh(Xs @ W1.T + b1) @ W2.T + b2)[:, 0]
    (V1, c1), (V2, c2) = params.g_layers  # type: ignore[misc]
    log_g = np.clip((np.tanh(Xs @ V1.T + c1) @ V2.T + c2)[:, 0], -12.0, 12.0)
    g = np.exp(log_g)
    eta = np.random.default_rng(3).standard_normal((5, 7))
    Y1 = f[:, None] + np.sqrt(g)[:, None] * eta
    (We, be) = params.eps_layers[0]  # type: ignore[misc]
    onehot = np.ones((5, 7, 1))  # T=1: the one-hot of t=1 is [1]
    inp = np.concatenate([Y1[..., None], onehot, np.broadcast_to(Xs[:, None, :], (5, 7, 2))], -1)
    eps_hat = (inp @ We.T + be)[..., 0]  # single linear layer, no hidden activation
    sched = params.schedule
    sab = sched.sqrt_alpha_bar[0]
    sbb = sched.sqrt_beta_bar[0]
    Y0 = (Y1 - (1.0 - sab) * f[:, None] - sbb * np.sqrt(g)[:, None] * eps_hat) / sab
    expected = params.y_mean + params.y_std * Y0  # clip inactive: |Y0| << 50
    assert float(np.max(np.abs(Y0))) < 50.0
    np.testing.assert_allclose(samples, expected, rtol=1e-12, atol=1e-12)


def test_handmade_inference_surface_and_fail_closed_edges() -> None:
    """Sampler determinism, quantile/PIT/CRPS wiring against the repo's
    scoring helpers, and the fail-closed inference surface — all without
    torch on hand-injected params."""
    model = _handmade(seed=1, n_steps=3)
    rng = np.random.default_rng(12)
    X = rng.normal(size=(9, 2))
    y = rng.normal(size=9)
    s1 = model.predict_samples(X, 40, seed=5)
    assert s1.shape == (9, 40)
    assert np.array_equal(s1, model.predict_samples(X, 40, seed=5))  # seeded
    assert not np.array_equal(s1, model.predict_samples(X, 40, seed=6))
    assert np.all(np.isfinite(s1))
    # quantile wiring: exactly the seeded-sample quantiles, non-crossing
    taus = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    q = model.predict_quantiles(X, taus, n_samples=40, seed=5)
    np.testing.assert_allclose(q, np.quantile(s1, taus, axis=1).T, rtol=1e-12, atol=1e-12)
    assert np.all(np.diff(q, axis=1) >= 0.0)
    # CRPS wiring: mean of the repo's empirical CRPS on the same seeded draws
    manual = float(np.mean([crps_empirical(float(y[i]), s1[i]) for i in range(y.size)]))
    assert model.crps(X, y, n_samples=40, seed=5) == pytest.approx(manual, rel=1e-12)
    assert model.crps(X, y, n_samples=40, seed=5) > 0.0
    # endpoint wiring: closed-form Gaussian CRPS of N(f, g)
    f, sigma = model.predict_params(X)
    assert np.all(sigma > 0.0) and np.all(np.isfinite(f))
    assert model.endpoint_crps(X, y) == pytest.approx(
        float(np.mean(crps_gaussian(y, f, sigma))), rel=1e-12
    )
    assert np.isfinite(model.endpoint_log_score(X, y))
    # PIT wiring: documented midrank empirical CDF, strictly inside (0, 1)
    pit = model.pit(X, y, n_samples=40, seed=5)
    less = np.sum(s1 < y[:, None], axis=1)
    np.testing.assert_allclose(pit, (less + 0.5) / 41.0, rtol=1e-12, atol=1e-12)
    assert np.all((pit > 0.0) & (pit < 1.0))
    # log_g clip is active and exact: inject a huge g-head bias
    (V1, c1), (V2, c2) = model._params.g_layers  # type: ignore[misc]
    clipped = dfm._NumpyDiffPTSParams(
        schedule=model._params.schedule,
        ablation="full",
        log_g_clip=12.0,
        y_mean=0.5,
        y_std=2.0,
        ctx_mean=np.zeros(2),
        ctx_std=np.ones(2),
        f_layers=model._params.f_layers,
        g_layers=((V1, c1 + 50.0), (V2, c2 + 50.0)),
        eps_layers=model._params.eps_layers,
        y0_clip_lo=-50.0,
        y0_clip_hi=50.0,
    )
    model._params = clipped
    _, sig_big = model.predict_params(X)
    # log g saturates at +12 -> g = e^12 -> sigma = y_std * sqrt(g) = 2 e^6
    np.testing.assert_allclose(sig_big, 2.0 * math.exp(6.0), rtol=1e-12)
    # fail-closed edges
    fresh = DiffPTSForecaster()
    with pytest.raises(RuntimeError, match="not fitted"):
        fresh.predict(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        fresh.predict_samples(X, 10)
    with pytest.raises(RuntimeError, match="not fitted"):
        fresh.crps(X, y)
    with pytest.raises(ValueError, match="feature count"):
        model.predict(X[:, :1])
    with pytest.raises(ValueError, match="2-D"):
        model.predict(X.ravel())
    with pytest.raises(ValueError, match="taus"):
        model.predict_quantiles(X, np.array([0.0, 0.5]))
    with pytest.raises(ValueError, match="taus"):
        model.predict_quantiles(X, np.array([1.0]))
    with pytest.raises(ValueError, match="taus"):
        model.predict_quantiles(X, np.array([np.nan]))
    with pytest.raises(ValueError, match="taus"):
        model.predict_quantiles(X, np.array([]))
    with pytest.raises(ValueError, match="n_samples"):
        model.predict_samples(X, 0)
    with pytest.raises(ValueError, match="length"):
        model.crps(X, y[:-1])
    with pytest.raises(ValueError, match="finite"):
        model.endpoint_crps(X, np.full(9, np.inf))


def test_constructor_fail_closed() -> None:
    for kw in (
        {"n_steps": 0},
        {"n_steps": True},
        {"schedule": "vsde"},
        {"weighting": "heavy"},
        {"ablation": "some"},
        {"hidden": ()},
        {"hidden": (0,)},
        {"hidden": (4, -2)},
        {"epochs": 0},
        {"lr": 0.0},
        {"lr": -1.0},
        {"batch_size": 0},
        {"batch_size": -3},
        {"batch_size": True},
        {"log_g_clip": 0.0},
        {"beta_start": -1.0},
    ):
        with pytest.raises(ValueError):
            DiffPTSForecaster(**kw)  # type: ignore[arg-type]


def test_fit_input_validation_runs_without_torch() -> None:
    """Input-contract errors raise ValueError BEFORE torch is touched, so the
    fail-closed surface holds (and is testable) even without the nn extra."""
    model = DiffPTSForecaster(epochs=2)
    rng = np.random.default_rng(6)
    X = rng.normal(size=(32, 2))
    y = rng.normal(size=32)
    with pytest.raises(ValueError, match="2-D"):
        model.fit(X.ravel(), y)
    with pytest.raises(ValueError, match="length"):
        model.fit(X, y[:-1])
    with pytest.raises(ValueError, match="finite"):
        model.fit(X, np.r_[np.nan, y[1:]])
    with pytest.raises(ValueError, match="too few"):
        model.fit(X[:4], y[:4])
    # constant series: fail closed, nothing to fit
    with pytest.raises(ValueError, match="zero variance"):
        model.fit(X[:20], np.full(20, 3.0))


def test_module_imports_without_torch_and_raises_clear_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No top-level torch import; the numpy core (schedules, Algorithm-2
    sampler) stays usable with torch blocked; every training entry point
    fails closed with nn-extra guidance."""
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    spec = importlib.util.spec_from_file_location("_diffpts_no_torch", dfm.__file__)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_diffpts_no_torch", probe)
    spec.loader.exec_module(probe)  # must import cleanly without torch
    # numpy core works while torch is blocked
    s = probe.noise_schedule(5, "cosine")
    assert s.n_steps == 5 and np.all(np.diff(s.alpha_bar) < 0.0)
    rng = np.random.default_rng(7)
    X, y = rng.normal(size=(16, 2)), rng.normal(size=16)
    with pytest.raises(ImportError, match=r"'nn' extra"):
        probe.DiffPTSForecaster(epochs=2).fit(X, y)
    with pytest.raises(ImportError, match=r"'nn' extra"):
        probe.bench_diffpts(n_train=10, n_test=6, epochs=2, ngboost_rounds=2, torf_epochs=2)


# ---------------------------------------------------------------------------
# torch lane (skipped cleanly when the nn extra is absent)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_gauss() -> dict[str, Any]:
    """DiffPTS and a CRPS-trained NGBoostGaussian on a SYNTHETIC seeded
    heteroskedastic GAUSSIAN stream: the LSNM Gaussian endpoint contains the
    truth, so sigma recovery and nominal calibration are testable."""
    if not _HAS_TORCH:
        pytest.skip("DiffPTS training requires the nn extra (torch)")
    Xtr, ytr, Xte, yte, gamma_te = _gauss_hetero_stream(1000, 600, seed=0)
    diffpts = DiffPTSForecaster(seed=0).fit(Xtr, ytr)
    ngb = NGBoostGaussian(n_estimators=80, learning_rate=0.1, score="crps", seed=0).fit(Xtr, ytr)
    return {
        "Xtr": Xtr,
        "ytr": ytr,
        "Xte": Xte,
        "yte": yte,
        "gamma_te": gamma_te,
        "diffpts": diffpts,
        "ngb": ngb,
    }


@pytest.fixture(scope="module")
def fitted_student_t() -> dict[str, Any]:
    """The SHARED seeded heavy-tailed stream of the TORF lane (bitwise
    identical fixture): DiffPTS vs NGBoostGaussian vs the TORF odd flow."""
    if not _HAS_TORCH:
        pytest.skip("DiffPTS training requires the nn extra (torch)")
    from quant_fund.models.odd_residual_flows import OddResidualFlow, TORFForecaster

    Xtr, ytr, Xte, yte = dfm._synthetic_hetero_stream(1000, 600, seed=0, noise="student_t")
    diffpts = DiffPTSForecaster(seed=0).fit(Xtr, ytr)
    ngb = NGBoostGaussian(n_estimators=80, learning_rate=0.1, score="crps", seed=0).fit(Xtr, ytr)
    torf = TORFForecaster(flow=OddResidualFlow(epochs=250, seed=0)).fit(Xtr, ytr)
    return {"Xte": Xte, "yte": yte, "diffpts": diffpts, "ngb": ngb, "torf": torf}


@pytest.fixture(scope="module")
def fitted_bimodal() -> dict[str, Any]:
    """Same comparison on the symmetric bimodal regime, where a Gaussian
    endpoint is structurally mis-shaped and the diffusion refinement must
    earn its CRPS (the paper's Table 3 'w/o denoise' story in reverse)."""
    if not _HAS_TORCH:
        pytest.skip("DiffPTS training requires the nn extra (torch)")
    from quant_fund.models.odd_residual_flows import OddResidualFlow, TORFForecaster

    Xtr, ytr, Xte, yte = dfm._synthetic_hetero_stream(1000, 600, seed=0, noise="bimodal")
    diffpts = DiffPTSForecaster(seed=0).fit(Xtr, ytr)
    ngb = NGBoostGaussian(n_estimators=80, learning_rate=0.1, score="crps", seed=0).fit(Xtr, ytr)
    torf = TORFForecaster(flow=OddResidualFlow(epochs=250, seed=0)).fit(Xtr, ytr)
    return {"Xte": Xte, "yte": yte, "diffpts": diffpts, "ngb": ngb, "torf": torf}


@requires_torch
def test_determinism_and_seed_sensitivity() -> None:
    Xtr, ytr, Xte, yte, _ = _gauss_hetero_stream(300, 100, seed=5)
    kw = dict(n_steps=5, epochs=60, hidden=(16, 16), batch_size=256, lr=3e-3)
    a = DiffPTSForecaster(seed=7, **kw).fit(Xtr, ytr)  # type: ignore[arg-type]
    b = DiffPTSForecaster(seed=7, **kw).fit(Xtr, ytr)  # type: ignore[arg-type]
    c = DiffPTSForecaster(seed=8, **kw).fit(Xtr, ytr)  # type: ignore[arg-type]
    Xs = Xte[:32]
    assert np.array_equal(a.predict_params(Xs)[0], b.predict_params(Xs)[0])
    assert np.array_equal(a.predict_samples(Xs, 24, seed=3), b.predict_samples(Xs, 24, seed=3))
    assert a.fit_info is not None and b.fit_info is not None
    assert a.fit_info.loss_curve == b.fit_info.loss_curve
    assert not np.array_equal(a.predict_samples(Xs, 24, seed=3), c.predict_samples(Xs, 24, seed=3))
    assert not np.array_equal(a.predict_samples(Xs, 24, seed=3), a.predict_samples(Xs, 24, seed=4))


@requires_torch
def test_joint_objective_decomposition_matches_prop_3_3_and_3_1_at_init() -> None:
    """Eq. 14 / Eq. 12 at the zero-init endpoint (f=0, g=1, log g=0):
    L_NLL(0) = (1/2) mean(ys^2) exactly for the uniform weighting, and
    (abar_T/2) mean(ys^2) for the exact Proposition 3.1 weighting (the log-g
    half vanishes at g=1); the total loss is the sum of the recorded parts,
    and the denoising term starts at the unpredictable-noise level ~1."""
    Xtr, ytr, _, _, _ = _gauss_hetero_stream(300, 50, seed=2)
    ys = (ytr - ytr.mean()) / ytr.std()
    half_ms = 0.5 * float(np.mean(ys**2))  # = 0.5 up to float32 casting
    uni = DiffPTSForecaster(
        n_steps=5, schedule="linear", epochs=1, hidden=(8, 8), batch_size=None, seed=0
    ).fit(Xtr, ytr)
    assert uni.fit_info is not None
    assert uni.fit_info.nll_curve[0] == pytest.approx(half_ms, abs=1e-3)
    assert 0.7 < uni.fit_info.denoise_curve[0] < 1.7  # eps net at random init
    assert uni.fit_info.loss_curve[0] == pytest.approx(
        uni.fit_info.denoise_curve[0] + uni.fit_info.nll_curve[0], abs=1e-5
    )
    sched = noise_schedule(5, "linear", 1e-4, 0.02)
    ex = DiffPTSForecaster(
        n_steps=5,
        schedule="linear",
        epochs=1,
        hidden=(8, 8),
        batch_size=None,
        weighting="elbo",
        seed=0,
    ).fit(Xtr, ytr)
    assert ex.fit_info is not None
    # Prop 3.1: quadratic endpoint term weighted by abar_T / 2
    assert ex.fit_info.nll_curve[0] == pytest.approx(sched.alpha_bar_T * half_ms, abs=1e-3)
    assert ex.fit_info.nll_curve[0] < uni.fit_info.nll_curve[0]  # abar_T < 1


@requires_torch
def test_ablation_structure_instantiates_remark_3_2_chain() -> None:
    """Table 3 / Remark 3.2 instantiations: f_phi == 0 (DDPM end of the
    chain), g_psi == 1 (RDIT/D3U end: the NLL reduces to MSE), and
    no_denoise (sampling directly from the endpoint N(f, g), the paper's
    'w/o denoise' evaluation protocol)."""
    Xtr, ytr, Xte, _ = dfm._synthetic_hetero_stream(400, 100, seed=3, noise="student_t")
    kw = dict(n_steps=5, epochs=40, hidden=(16, 16), batch_size=256, lr=3e-3, seed=0)
    nof = DiffPTSForecaster(ablation="no_nll_f", **kw).fit(Xtr, ytr)  # type: ignore[arg-type]
    assert nof._params is not None and nof._params.f_layers is None
    mu = nof.predict(Xte)  # f == 0 standardized -> the constant train mean
    np.testing.assert_allclose(mu, np.full(mu.shape, float(np.mean(ytr))), rtol=1e-5)
    nog = DiffPTSForecaster(ablation="no_nll_g", **kw).fit(Xtr, ytr)  # type: ignore[arg-type]
    assert nog._params is not None and nog._params.g_layers is None
    sigma = nog.predict_sigma(Xte)  # g == 1 standardized -> sigma == y_std
    np.testing.assert_allclose(sigma, np.full(sigma.shape, float(np.std(ytr))), rtol=1e-5)
    nod = DiffPTSForecaster(ablation="no_denoise", **kw).fit(Xtr, ytr)  # type: ignore[arg-type]
    assert nod._params is not None and nod._params.eps_layers is None
    f, sg = nod.predict_params(Xte)
    eta = np.random.default_rng(9).standard_normal((Xte.shape[0], 30))
    np.testing.assert_allclose(
        nod.predict_samples(Xte, 30, seed=9), f[:, None] + sg[:, None] * eta, rtol=1e-12
    )
    # sample CRPS of the endpoint sampler ~= closed-form endpoint CRPS (MC)
    y0 = np.zeros(Xte.shape[0])
    assert nod.crps(Xte, y0, n_samples=300, seed=9) == pytest.approx(
        nod.endpoint_crps(Xte, y0), abs=0.06
    )


@requires_torch
def test_noiseless_degenerate_scale_collapse() -> None:
    """Degenerate check: a deterministic target (y = mu(x) exactly, zero
    conditional noise) must drive the LSNM scale toward the floor — the
    Gaussian NLL half of Eq. 14 is minimized at g -> 0 when f fits exactly.
    Known honest limitation: the SAMPLE-based CRPS degrades here because the
    x0-hat inversion (Algorithm 2 step 5) amplifies residual denoiser error
    by sqrt(beta_bar_t g)/sqrt(abar_t) at large t; the endpoint scores are
    the meaningful degenerate evidence."""
    rng = np.random.default_rng(0)
    n = 400
    x0 = rng.uniform(-1.0, 1.0, n)
    x1 = rng.uniform(-1.0, 1.0, n)
    feats = np.column_stack([x0, x1, x1 * x1])
    y = 2.0 * x0 + x1 * x1  # exactly deterministic given the features
    model = DiffPTSForecaster(
        n_steps=10, epochs=300, lr=3e-3, hidden=(32, 32), batch_size=200, seed=0
    ).fit(feats[:300], y[:300])
    y_std = float(np.std(y[:300]))
    sigma = model.predict_sigma(feats[300:])
    assert float(np.max(sigma)) < 0.2 * y_std  # near-zero scale (reference: 0.11)
    assert model.endpoint_crps(feats[300:], y[300:]) < 0.02 * y_std
    assert np.all(np.isfinite(model.fit_info.loss_curve))  # type: ignore[union-attr]
    # sampled CRPS stays finite and bounded (documented degradation, above)
    assert model.crps(feats[300:], y[300:], n_samples=100, seed=1) < 0.5 * y_std


@requires_torch
def test_gaussian_recovery_calibration_and_crps_dominance(fitted_gauss: dict[str, Any]) -> None:
    """On the correctly-specified Gaussian stream: sigma tracks the true
    gamma(x), the diffusion CRPS beats NGBoostGaussian under the SAME proper
    score, PIT/coverage are sane at the documented MC floor, and the sample
    CRPS agrees with the closed-form endpoint CRPS within MC noise (the
    diffusion refines an already-correct Gaussian only marginally).

    Reference margins (seed 0, M=200): NGBoost +2.8e-2; cov90 = 0.863;
    PIT-KS = 0.043 (floor ~0.032), p = 0.21.
    """
    d: DiffPTSForecaster = fitted_gauss["diffpts"]
    ngb: NGBoostGaussian = fitted_gauss["ngb"]
    Xte, yte = fitted_gauss["Xte"], fitted_gauss["yte"]
    gamma_te = fitted_gauss["gamma_te"]
    mu_true = 2.0 * Xte[:, 0] + Xte[:, 1] ** 2
    d_crps = d.crps(Xte, yte, n_samples=200, seed=1)
    ngb_crps = float(ngb.crps(Xte, yte))
    assert d_crps < ngb_crps
    assert d_crps <= ngb_crps - 0.01  # margin >= 3x the MC+float slack
    # LSNM backbone recovery: sigma tracks gamma(x), f tracks mu(x)
    f, sigma = d.predict_params(Xte)
    assert float(np.corrcoef(sigma, gamma_te)[0, 1]) > 0.85
    assert float(np.corrcoef(f, mu_true)[0, 1]) > 0.95
    assert float(np.mean(np.abs(f - mu_true))) < 0.15
    # closed-form endpoint vs sample CRPS: consistent within MC tolerance
    assert abs(d_crps - d.endpoint_crps(Xte, yte)) < 0.03
    # calibration sanity (documented MC floor on the KS statistic)
    q = d.predict_quantiles(Xte, np.array([0.05, 0.95]), n_samples=200, seed=1)
    cov = float(np.mean((yte >= q[:, 0]) & (yte <= q[:, 1])))
    assert 0.78 <= cov <= 0.94  # reference 0.863; mild reverse-pass shrinkage
    ks, p = pit_ks(d.pit(Xte, yte, n_samples=200, seed=1))
    assert ks < PIT_KS_FLOOR + 0.06  # reference 0.043
    assert p > 0.005  # reference 0.21
    # proper-score-only sanity: endpoint log score finite; MAE tracks NGBoost's
    assert np.isfinite(d.endpoint_log_score(Xte, yte))
    assert d.mae(Xte, yte) < float(np.mean(np.abs(yte - ngb.predict(Xte)))) + 0.05


@requires_torch
def test_student_t_beats_ngboost_and_matches_torf(fitted_student_t: dict[str, Any]) -> None:
    """The paper's claim on the SHARED seeded heavy-tailed stream: full-ELBO
    DiffPTS CRPS <= NGBoostGaussian CRPS under the same proper score, and
    competitive with the TORF odd residual flow (a flexible non-Gaussian
    density baseline trained by exact flow NLL).

    Reference margins (seed 0, M=200): NGBoost +1.3e-2, TORF -0.0e-2 (a
    statistical tie); cov90 = 0.848; PIT p = 0.25.
    """
    d: DiffPTSForecaster = fitted_student_t["diffpts"]
    ngb: NGBoostGaussian = fitted_student_t["ngb"]
    torf = fitted_student_t["torf"]
    Xte, yte = fitted_student_t["Xte"], fitted_student_t["yte"]
    d_crps = d.crps(Xte, yte, n_samples=200, seed=1)
    ngb_crps = float(ngb.crps(Xte, yte))
    torf_crps = float(torf.crps(Xte, yte))
    assert d_crps < ngb_crps
    assert d_crps <= ngb_crps - 0.005  # reference margin 0.013
    assert d_crps <= torf_crps + 0.012  # reference: dead heat (-0.0002)
    assert d_crps <= d.endpoint_crps(Xte, yte) + CRPS_TOL  # refinement never hurts here
    q = d.predict_quantiles(Xte, np.array([0.05, 0.95]), n_samples=200, seed=1)
    cov = float(np.mean((yte >= q[:, 0]) & (yte <= q[:, 1])))
    assert 0.78 <= cov <= 0.93  # reference 0.848
    ks, p = pit_ks(d.pit(Xte, yte, n_samples=200, seed=1))
    assert ks < PIT_KS_FLOOR + 0.06
    assert p > 0.01  # reference 0.25


@requires_torch
def test_bimodal_beats_ngboost_and_own_endpoint_competes_with_torf(
    fitted_bimodal: dict[str, Any],
) -> None:
    """Symmetric bimodal residuals: the Gaussian endpoint is structurally
    mis-shaped, so the diffusion refinement must earn the score — it beats
    its OWN closed-form endpoint by >= 1.5e-2 (the paper's Table 3 'w/o
    denoise' comparison) and NGBoost by a wide margin, and stays competitive
    with the TORF spline flow.

    Reference margins (seed 0, M=200): NGBoost +8.0e-2, endpoint +4.4e-2,
    TORF +1.0e-2; cov90 = 0.937. Honest calibration note: strict PIT
    uniformity is REJECTED here (KS ~ 0.16 vs the 0.032 MC floor) — the
    0.15*gamma-wide clumps are smoothed by the tiny denoiser; CRPS, the
    paper's headline metric, dominates regardless.
    """
    d: DiffPTSForecaster = fitted_bimodal["diffpts"]
    ngb: NGBoostGaussian = fitted_bimodal["ngb"]
    torf = fitted_bimodal["torf"]
    Xte, yte = fitted_bimodal["Xte"], fitted_bimodal["yte"]
    d_crps = d.crps(Xte, yte, n_samples=200, seed=1)
    ngb_crps = float(ngb.crps(Xte, yte))
    torf_crps = float(torf.crps(Xte, yte))
    endpoint_crps = d.endpoint_crps(Xte, yte)
    assert d_crps <= ngb_crps - 0.04  # reference margin 0.080
    assert d_crps <= endpoint_crps - 0.015  # reference 0.044: denoising earns its keep
    assert d_crps <= torf_crps + 0.03  # reference +0.010
    q = d.predict_quantiles(Xte, np.array([0.05, 0.5, 0.95]), n_samples=200, seed=1)
    assert np.all(np.diff(q, axis=1) >= 0.0)
    cov = float(np.mean((yte >= q[:, 0]) & (yte <= q[:, 2])))
    assert 0.72 <= cov <= 0.98  # reference 0.937 (config-sensitive; wide honest band)
    ks, _ = pit_ks(d.pit(Xte, yte, n_samples=200, seed=1))
    assert ks < 0.22  # reference 0.160 — clump smoothing, documented above


@requires_torch
def test_bench_keys_labels_and_claims() -> None:
    """bench_diffpts: scorecard-ready proper-score keys, honesty labels, and
    DiffPTS >= NGBoost on CRPS over the shared SYNTHETIC fixture DGP — never
    market evidence."""
    out = bench_diffpts(
        n_train=700,
        n_test=400,
        seed=0,
        epochs=200,
        n_samples=200,
        ngboost_rounds=60,
        torf_epochs=200,
    )
    expected = {
        "synthetic_diffpts_crps",
        "synthetic_diffpts_endpoint_crps",
        "synthetic_ngboost_crps",
        "synthetic_torf_crps",
        "synthetic_crps_gain_vs_ngboost",
        "synthetic_crps_gain_vs_torf",
        "synthetic_diffpts_endpoint_log_score",
        "synthetic_ngboost_log_score",
        "synthetic_diffpts_mae",
        "synthetic_sample_mean_dev",
        "synthetic_coverage_90",
        "synthetic_pit_ks",
        "synthetic_pit_ks_pvalue",
        "synthetic_n_train",
        "synthetic_n_test",
        "synthetic_seed",
        "synthetic_noise",
        "synthetic_schedule",
        "synthetic_dgp",
        "synthetic_claim",
        "synthetic_synthetic",
    }
    assert set(out) == expected
    assert out["synthetic_dgp"] == "fixture"
    assert out["synthetic_claim"] == "research_metric_only"
    assert out["synthetic_synthetic"] == "heteroskedastic_symmetric_seeded"
    assert out["synthetic_crps_gain_vs_ngboost"] > 0.0  # reference +0.023
    assert float(out["synthetic_diffpts_crps"]) <= float(out["synthetic_ngboost_crps"]) + CRPS_TOL
    assert (
        abs(
            float(out["synthetic_crps_gain_vs_ngboost"])
            - (float(out["synthetic_ngboost_crps"]) - float(out["synthetic_diffpts_crps"]))
        )
        < 1e-12
    )
    assert 0.75 <= float(out["synthetic_coverage_90"]) <= 0.96
    assert abs(float(out["synthetic_sample_mean_dev"])) <= 0.1  # reference 0.06
    for key, value in out.items():
        if isinstance(value, float):
            assert math.isfinite(value), key
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav", "drawdown")
    assert not any(tok in key.lower() for key in out for tok in forbidden)
