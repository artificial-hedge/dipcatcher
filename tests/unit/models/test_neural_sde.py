"""Tests for quant_fund.models.neural_sde — latent-SDE probabilistic forecaster.

Coverage map:

* closed-form equalities — softmax-free softplus inverse, KL(N||N(0,I)),
  Girsanov path-KL with constant correction (exact, rtol 1e-12), EM solver
  marginal law under zero drift (pure BM: N(z0, s0^2 + k sigma^2 dt)) and
  under an affine OU drift (AR(1) with a = 1 - theta dt; mean and variance
  closed forms), decoder identity at zero init, logsignature feature counts
  (Witt numbers on 2 letters).
* seeded SYNTHETIC Monte-Carlo bands — EM weak convergence vs the exact
  AR(1) law (4*MC-se slack), simulator stationary variances (15% rel),
  bench coverage/calibration bands at >= 6x binomial se slack.
* fail-closed edges — every validator and every predict-time guard raises
  ValueError on degenerate/invalid input; input errors in ``fit`` raise
  before torch is touched (the nn extra is only needed to actually train).
* determinism — same seed gives bit-identical fits and samples.
* torch-gated path — ``requires_torch`` skip pattern mirrors
  ``tests/unit/models/test_deep_hedging.py`` / ``test_diffusion_forecaster``;
  the module itself imports cleanly without the extra (checked by exec'ing
  the file with ``sys.modules['torch'] = None``).

Handmade-parameter injection (``model._params = _NumpyLatentSDEParams(...)``)
exercises the whole numpy inference path — encoder, q_0, EM rollouts under
prior/posterior drifts, decoder — without needing torch at test time. This
mirrors the diffusion_forecaster test template.

Spec-vs-paper corrections verified for the module docstring: the anchors
are Kidger et al. 2020 (arXiv:2005.08926, Neural CDEs), Kidger et al. 2021
(arXiv:2102.03657, Neural SDEs as Infinite-Dimensional GANs — the spec's
2002.09329 resolves to an unrelated physics paper), and Li et al. 2020
(arXiv:2001.01328, Scalable Gradients for SDEs / "Latent SDE" — the spec's
2001.04328 resolves to a Kodaira-dimension paper and misnames the authors).
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from quant_fund.models import neural_sde as nsd

MODULE_PATH = Path(nsd.__file__).resolve()

# ---------------------------------------------------------------------------
# torch-gating (same pattern as test_deep_hedging.py / test_diffusion_forecaster)
# ---------------------------------------------------------------------------

_HAS_TORCH = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch (nn extra) not installed")


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------

_C = 8
_H = 4
_T = _C + _H
_D = 2
_CTX = 6
_SIG = 2  # logsignature order -> feature dim 3 + 3 scalars = 6


def _feat_dim(sig_order: int = _SIG) -> int:
    return nsd._sig_feature_count(sig_order) + 3


def _toy_params(
    *,
    context_len: int = _C,
    horizon: int = _H,
    latent_dim: int = _D,
    sig_order: int = _SIG,
    ctx_dim: int = _CTX,
    dt: float | None = None,
    sigma: float = 0.5,
    sigma_floor: float = 1e-3,
    y_mean: float = 0.0,
    y_std: float = 1.0,
    q0_bias: np.ndarray | None = None,
    u_bias: np.ndarray | None = None,
    f_layer: tuple[np.ndarray, np.ndarray] | None = None,
) -> nsd._NumpyLatentSDEParams:
    """Handmade inference params: enc -> ctx = 0; q_0 = N(q0_bias[:d], I);

    f(z, t) zero or a caller-supplied affine layer; u(z, t, ctx) = u_bias
    constant; sigma(z, t) = sigma constant; decoder identity N(0, 1) in
    standardized units (mu_raw = y_mean, sig_raw = y_std).
    """
    d, cd = latent_dim, ctx_dim
    dt = (1.0 / (context_len + horizon - 1)) if dt is None else dt
    F = _feat_dim(sig_order)
    enc_w = np.zeros((cd, F))
    enc_b = np.zeros(cd)
    q0_w = np.zeros((2 * d, cd))
    q0_b = np.zeros(2 * d) if q0_bias is None else np.asarray(q0_bias, dtype=float)
    f = (
        (np.zeros((d, d + 1)), np.zeros(d))
        if f_layer is None
        else (np.asarray(f_layer[0], dtype=float), np.asarray(f_layer[1], dtype=float))
    )
    u_w = np.zeros((d, d + 1 + cd))
    u_b = np.zeros(d) if u_bias is None else np.asarray(u_bias, dtype=float)
    sp = math.log(math.expm1(sigma - sigma_floor))
    sig_w = np.zeros((d, d + 1))
    sig_b = np.full(d, sp)
    dec_w = np.zeros((2, d + 1))
    dec_b = np.zeros(2)
    return nsd._NumpyLatentSDEParams(
        context_len=context_len,
        horizon=horizon,
        latent_dim=d,
        sig_order=sig_order,
        dt=dt,
        sigma_floor=sigma_floor,
        log_s_clip=nsd._LOG_S_CLIP,
        y_mean=y_mean,
        y_std=y_std,
        feat_mean=np.zeros(F),
        feat_std=np.ones(F),
        enc_layers=((enc_w, enc_b),),
        q0_layer=(q0_w, q0_b),
        f_layers=(f,),
        u_layers=((u_w, u_b),),
        sig_layers=((sig_w, sig_b),),
        dec_layers=((dec_w, dec_b),),
    )


def _toy_model(params: nsd._NumpyLatentSDEParams) -> nsd.NeuralSDEForecaster:
    """NeuralSDEForecaster with injected numpy params — no torch needed."""
    m = nsd.NeuralSDEForecaster(
        context_len=params.context_len,
        horizon=params.horizon,
        latent_dim=params.latent_dim,
        sig_order=params.sig_order,
        ctx_dim=params.enc_layers[0][0].shape[0],
    )
    m._params = params
    return m


def _rng_paths(n: int = 32, T: int = _T, seed: int = 0) -> np.ndarray:
    """Deterministic random-walk fixture (SYNTHETIC; finite, non-degenerate)."""
    rng = np.random.default_rng(seed)
    x = np.empty((n, T))
    x[:, 0] = rng.standard_normal(n)
    x[:, 1:] = x[:, :1] + np.cumsum(0.1 * rng.standard_normal((n, T - 1)), axis=1)
    return x


# ---------------------------------------------------------------------------
# module import without torch (the lazy _torch() contract)
# ---------------------------------------------------------------------------


def test_module_imports_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The module must import with torch absent (lazy ``_torch()`` pattern)."""
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    spec = importlib.util.spec_from_file_location("_nsde_no_torch", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_nsde_no_torch", mod)  # dataclasses resolve via sys.modules
    spec.loader.exec_module(mod)
    assert mod.NeuralSDEForecaster is not None
    with pytest.raises(ImportError):
        mod._torch()


def test_torch_helper_returns_module_when_present() -> None:
    if not _HAS_TORCH:
        pytest.skip("torch absent")
    assert nsd._torch().__name__ == "torch"


def test_docstring_documents_citation_corrections() -> None:
    doc = nsd.__doc__ or ""
    assert "arXiv:2001.01328" in doc  # Latent SDE (spec's 2001.04328 was wrong)
    assert "arXiv:2102.03657" in doc  # Neural SDE GAN (spec's 2002.09329 was wrong)
    assert "Honesty" in doc


# ---------------------------------------------------------------------------
# validators (fail-closed; pure numpy)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(5),  # 1-D
        np.zeros((0, 5)),  # no paths
        np.zeros((3, 1)),  # T < 2
        np.full((4, 6), np.nan),  # NaN
        np.full((4, 6), np.inf),  # inf
    ],
)
def test_check_paths_fail_closed(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        nsd._check_paths(bad)


@pytest.mark.parametrize(
    "bad_ctx",
    [
        np.zeros((3, _C + 1)),  # wrong width
        np.zeros((0, _C)),  # empty
        np.full((3, _C), np.nan),
    ],
)
def test_check_contexts_fail_closed(bad_ctx: np.ndarray) -> None:
    with pytest.raises(ValueError):
        nsd._check_contexts(bad_ctx, _C)


def test_check_contexts_1d_promotes_to_row() -> None:
    out = nsd._check_contexts(np.zeros(_C), _C)
    assert out.shape == (1, _C)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "3"])
def test_check_count_fail_closed(bad: object) -> None:
    with pytest.raises(ValueError):
        nsd._check_count(bad, "x")


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_check_positive_fail_closed(bad: float) -> None:
    with pytest.raises(ValueError):
        nsd._check_positive(bad, "x")


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_check_nonnegative_fail_closed(bad: float) -> None:
    with pytest.raises(ValueError):
        nsd._check_nonnegative(bad, "x")


def test_check_choice_fail_closed() -> None:
    with pytest.raises(ValueError):
        nsd._check_choice("milstein", nsd._DRIFTS, "drift")


@pytest.mark.parametrize(
    "kw",
    [
        {"context_len": 1},
        {"context_len": 2.5},
        {"horizon": 0},
        {"latent_dim": 0},
        {"sig_order": 0},
        {"sig_order": 5},
        {"ctx_dim": 0},
        {"hidden": ()},
        {"hidden": (8, -1)},
        {"epochs": 0},
        {"batch_size": 0},
        {"sigma_floor": -1.0},
        {"log_s_clip": 0.0},
        {"kl_weight": -0.5},
        {"lr": 0.0},
    ],
)
def test_ctor_fail_closed(kw: dict) -> None:
    with pytest.raises(ValueError):
        nsd.NeuralSDEForecaster(**kw)


# ---------------------------------------------------------------------------
# numpy primitive closed forms
# ---------------------------------------------------------------------------


def test_sig_feature_count_witt_numbers() -> None:
    # Lyndon/Witt counts on a 2-D (t, x) path: cumulative coords to order k.
    assert nsd._sig_feature_count(1) == 2
    assert nsd._sig_feature_count(2) == 3
    assert nsd._sig_feature_count(3) == 5
    assert nsd._sig_feature_count(4) == 8


def test_context_features_shape_and_determinism() -> None:
    ctx = _rng_paths(16, _C, seed=1)
    f1 = nsd._context_features(ctx, _C, 0.1, _SIG)
    f2 = nsd._context_features(ctx, _C, 0.1, _SIG)
    assert f1.shape == (16, _feat_dim(_SIG))
    assert np.array_equal(f1, f2)
    assert np.all(np.isfinite(f1))


def test_softplus_inverse_consistency() -> None:
    # inverse-softplus log(expm1(x)) is defined on x > 0 — the only domain the
    # module uses it on (sigma init / sigma_floor are strictly positive).
    x = np.linspace(0.01, 6.0, 13)
    assert np.allclose(nsd._softplus(np.log(np.expm1(x))), x, rtol=1e-12, atol=1e-12)
    assert nsd._softplus(np.array([0.0]))[0] == pytest.approx(math.log(2.0))


def test_kl_normal_closed_form() -> None:
    mu = np.array([[0.0, 1.0], [2.0, -1.0]])
    log_s = np.array([[0.0, 0.5], [-0.25, 1.0]])
    expect = 0.5 * np.sum(mu**2 + np.exp(2 * log_s) - 1.0 - 2 * log_s, axis=-1)
    out = nsd._kl_normal(mu, log_s)
    np.testing.assert_allclose(out, expect, rtol=1e-14)


def test_gaussian_logpdf_closed_form() -> None:
    y = np.array([0.0, 1.5, -2.0])
    m = np.array([0.5, 0.0, -1.0])
    s = np.array([1.0, 2.0, 0.5])
    log_s = np.log(s)
    expect = -0.5 * ((y - m) / s) ** 2 - np.log(s) - 0.5 * math.log(2.0 * math.pi)
    np.testing.assert_allclose(nsd._gaussian_logpdf(y, m, log_s), expect, rtol=1e-14)


def test_np_mlp_matches_affine_tanh_chain() -> None:
    W1 = np.array([[1.0, -1.0], [0.5, 0.5]])
    b1 = np.array([0.1, -0.2])
    W2 = np.array([[1.0, 1.0]])
    b2 = np.array([0.3])
    x = np.array([[0.2, -0.4], [1.0, 1.0]])
    expect = np.tanh(x @ W1.T + b1) @ W2.T + b2
    np.testing.assert_allclose(nsd._np_mlp(x, ((W1, b1), (W2, b2))), expect, rtol=1e-14)


def test_mixture_quantiles_single_gaussian_exact() -> None:
    from scipy.stats import norm

    taus = (0.05, 0.5, 0.95)
    out = nsd._mixture_quantiles(np.array([1.0]), np.array([0.3]), np.array([0.7]), taus)
    expect = 0.3 + 0.7 * norm.ppf(np.asarray(taus))
    np.testing.assert_allclose(out, expect, atol=5e-3)  # grid resolution bound


def test_mixture_quantiles_two_component_brackets() -> None:
    # equal-weight bimodal with s << |mu|: the 25% quantile is the 50th
    # percentile of the LEFT component (exact: -m), the 75% the right's (+m),
    # and the tails the Gaussian 10% points of each lobe (~grid resolution).
    m, s = 2.0, 0.2
    w = np.array([0.5, 0.5])
    mu = np.array([-m, m])
    sig = np.array([s, s])
    q05, q25, q75, q95 = nsd._mixture_quantiles(w, mu, sig, (0.05, 0.25, 0.75, 0.95))
    from scipy.stats import norm

    z10 = float(norm.ppf(0.1))
    np.testing.assert_allclose([q25, q75], [-m, m], atol=5e-3)
    np.testing.assert_allclose([q05, q95], [-m + s * z10, m - s * z10], atol=5e-3)


# ---------------------------------------------------------------------------
# Girsanov path-KL and EM rollout closed forms (handmade params, no torch)
# ---------------------------------------------------------------------------


def test_girsanov_kl_zero_correction() -> None:
    p = _toy_params()
    z_path = np.zeros((3, _T, _D))
    kl = nsd._girsanov_kl_path(p, z_path, np.zeros((3, _CTX)))
    np.testing.assert_allclose(kl, 0.0, atol=0.0)


def test_girsanov_kl_constant_correction_exact() -> None:
    # u = c const, sigma = s const -> KL = (T-1) * ||c||^2 / (2 s^2) * dt,
    # path-independent: exact to fp precision on ANY latent path.
    c = np.array([0.4, -0.3])
    s = 0.5
    p = _toy_params(sigma=s, u_bias=c)
    rng = np.random.default_rng(7)
    z_path = rng.standard_normal((5, _T, _D))
    kl = nsd._girsanov_kl_path(p, z_path, np.zeros((5, _CTX)))  # noqa: E501
    expect = (_T - 1) * 0.5 * float(c @ c) / (s * s) * p.dt
    np.testing.assert_allclose(kl, np.full(5, expect), rtol=1e-12)


def test_girsanov_kl_short_path_raises() -> None:
    p = _toy_params()
    with pytest.raises(ValueError):
        nsd._girsanov_kl_path(p, np.zeros((2, 1, _D)), np.zeros((2, _CTX)))


def test_rollout_pure_bm_marginal_law() -> None:
    # f = u = 0, sigma const, q0 = N(v, I): z_k ~ N(v, I + k sigma^2 dt I).
    # Seeded MC check at 4*se slack (M = 4096) — SYNTHETIC.
    v = np.array([1.0, -0.5])
    s = 0.6
    dt = 0.25
    p = _toy_params(dt=dt, sigma=s, q0_bias=np.concatenate([v, np.zeros(_D)]))
    contexts = np.zeros((1, _C))
    z_h, _mu, _sig, _eta = nsd._np_rollout(p, contexts, _H, 4096, 1234, "posterior")
    for j in range(_H):
        k = _C + j  # z_h[:, j] lands at grid C+j after C+j EM steps
        var_expect = 1.0 + k * s * s * dt
        samp_mean = z_h[0, j].mean(axis=0)
        samp_var = z_h[0, j].var(axis=0, ddof=1)
        se_mean = math.sqrt(var_expect / 4096)
        np.testing.assert_allclose(samp_mean, v, atol=4 * se_mean + 1e-3)
        np.testing.assert_allclose(samp_var, var_expect, rtol=0.10)


def test_rollout_ou_affine_drift_exact_ar1() -> None:
    # f(z, t) = theta (mu - z) is AFFINE in z: the EM recursion is the AR(1)
    # z_{k+1} - mu = a (z_k - mu) + sigma sqrt(dt) eps, a = 1 - theta dt.
    # Mean: mu + (v - mu) a^k. Var: s0^2 a^{2k} + sigma^2 dt (1 - a^{2k})/(1 - a^2).
    theta, mu_ou, s = 2.0, 0.3, 0.5
    dt = 0.2
    W = np.zeros((_D, _D + 1))
    W[:, :_D] = -theta * np.eye(_D)
    b = np.full(_D, theta * mu_ou)
    v = np.array([1.5, -0.8])
    s0 = 1e-4  # near-deterministic posterior init at v
    q0_bias = np.concatenate([v, np.full(_D, math.log(s0))])
    p = _toy_params(dt=dt, sigma=s, q0_bias=q0_bias, f_layer=(W, b))
    a = 1.0 - theta * dt
    z_h, _m, _sg, _e = nsd._np_rollout(p, np.zeros((1, _C)), _H, 4096, 99, "posterior")
    for j in range(_H):
        k = _C + j
        mean_expect = mu_ou + (v - mu_ou) * a**k
        var_expect = s0**2 * a ** (2 * k) + s * s * dt * (1.0 - a ** (2 * k)) / (1.0 - a * a)
        samp_mean = z_h[0, j].mean(axis=0)
        samp_var = z_h[0, j].var(axis=0, ddof=1)
        np.testing.assert_allclose(
            samp_mean, mean_expect, atol=4 * math.sqrt(var_expect / 4096) + 1e-3
        )
        np.testing.assert_allclose(samp_var, var_expect, rtol=0.10)


def test_rollout_determinism_same_seed() -> None:
    p = _toy_params()
    ctx = np.zeros((2, _C))
    a = nsd._np_rollout(p, ctx, _H, 64, 5, "posterior")
    b = nsd._np_rollout(p, ctx, _H, 64, 5, "posterior")
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_rollout_prior_vs_posterior_drift() -> None:
    # Constant u = c: posterior horizon adds exactly +c*dt per step on top of
    # the prior rollout — SAME seeded draws for eps0/filter/horizon noise, so
    # the difference is exact (not statistical): zh_post[:, j] - zh_prior[:, j]
    # = c * (j + 1) * dt elementwise.
    c = np.array([0.8, 0.8])
    p = _toy_params(sigma=0.3, u_bias=c)
    ctx = np.zeros((2, _C))
    zh_post, *_ = nsd._np_rollout(p, ctx, _H, 64, 5, "posterior")
    zh_prior, *_ = nsd._np_rollout(p, ctx, _H, 64, 5, "prior")
    for j in range(_H):
        np.testing.assert_allclose(
            zh_post[:, j] - zh_prior[:, j],
            np.broadcast_to(c * (j + 1) * p.dt, zh_post[:, j].shape),
            rtol=1e-12,
            atol=1e-12,
        )


def test_decode_identity_at_zero_init() -> None:
    # zero decoder -> standardized N(0, 1): raw (mu_raw, sig_raw) = (y_mean, y_std).
    y_mean, y_std = 1.7, 0.4
    p = _toy_params(y_mean=y_mean, y_std=y_std)
    m = _toy_model(p)
    mu, sig = m.predict_components(np.zeros((3, _C)), n_samples=16, seed=0)
    np.testing.assert_allclose(mu, y_mean, rtol=0, atol=0)
    np.testing.assert_allclose(sig, y_std, rtol=0, atol=0)


def test_predict_samples_are_decode_noised_mixture() -> None:
    # zero-init: samples = y_mean + y_std * eta, eta iid N(0,1) — mean/sd
    # checks with seeded MC tolerance (M = 2048).
    y_mean, y_std = -0.5, 0.7
    p = _toy_params(y_mean=y_mean, y_std=y_std)
    m = _toy_model(p)
    s = m.predict_samples(np.zeros((4, _C)), n_samples=2048, seed=3)
    assert s.shape == (4, _H, 2048)
    np.testing.assert_allclose(s.mean(axis=2), y_mean, atol=4 * y_std / math.sqrt(2048) + 1e-3)
    np.testing.assert_allclose(s.std(axis=2, ddof=1), y_std, rtol=0.05)


def test_predict_before_fit_raises() -> None:
    m = nsd.NeuralSDEForecaster(context_len=_C, horizon=_H)
    with pytest.raises(RuntimeError):
        m.predict_samples(np.zeros((2, _C)))


@pytest.mark.parametrize(
    "call,kw",
    [
        ("predict_samples", {"n_ahead": 0}),
        ("predict_samples", {"n_samples": 0}),
        ("predict_samples", {"drift": "milstein"}),
        ("predict_components", {"n_samples": -2}),
        ("predict_quantiles", {"taus": []}),
        ("predict_quantiles", {"taus": [0.0, 0.5]}),
        ("predict_quantiles", {"taus": [0.5, 1.0]}),
        ("predict_quantiles", {"taus": [0.5, np.nan]}),
    ],
)
def test_predict_fail_closed(call: str, kw: dict) -> None:
    m = _toy_model(_toy_params())
    ctx = np.zeros((2, _C))
    base = {"contexts": ctx, "n_samples": 8, "seed": 0}
    base.update(kw)
    with pytest.raises(ValueError):
        getattr(m, call)(**base)


def test_predict_context_width_fail_closed() -> None:
    m = _toy_model(_toy_params())
    with pytest.raises(ValueError):
        m.predict_samples(np.zeros((2, _C + 1)))


def test_pit_in_unit_interval_and_shape() -> None:
    m = _toy_model(_toy_params(y_std=0.5))
    ctx = np.zeros((5, _C))
    pits = m.pit(ctx, np.zeros((5, _H)), n_samples=64, seed=1)
    assert pits.shape == (5, _H)
    assert np.all(pits > 0.0) and np.all(pits < 1.0)


def test_pit_at_zero_is_median_like() -> None:
    # zero-init decode -> predictive symmetric about y_mean: PIT(y_mean) ~ 0.5.
    p = _toy_params(y_mean=2.0, y_std=0.3)
    m = _toy_model(p)
    pits = m.pit(np.zeros((3, _C)), np.full((3, _H), 2.0), n_samples=512, seed=2)
    np.testing.assert_allclose(pits, 0.5, atol=0.05)


def test_predict_quantiles_monotone_and_shapes() -> None:
    m = _toy_model(_toy_params())
    q = m.predict_quantiles(np.zeros((3, _C)), np.array([0.05, 0.5, 0.95]), n_samples=256, seed=0)
    assert q.shape == (3, _H, 3)
    assert np.all(q[:, :, 0] <= q[:, :, 1]) and np.all(q[:, :, 1] <= q[:, :, 2])


def test_crps_matches_gaussian_closed_form_at_zero_init() -> None:
    # zero-init predictive is exactly N(y_mean, y_std): crps_empirical on the
    # seeded samples must land inside MC slack of crps_gaussian's closed form.
    from quant_fund.metrics.scoring import crps_gaussian

    y_mean, y_std = 0.2, 0.8
    m = _toy_model(_toy_params(y_mean=y_mean, y_std=y_std))
    ctx = np.zeros((4, _C))
    y = np.array([[0.1, -0.3, 0.6, 0.0]] * 4)
    s_hat = m.crps(ctx, y, n_samples=4096, seed=11)
    s_exact = float(np.mean(crps_gaussian(y.ravel(), np.full(16, y_mean), np.full(16, y_std))))
    # seeded MC slack: measured |s_hat - s_exact| = 0.006 at this seed; 10x.
    assert abs(s_hat - s_exact) < 0.06


def test_mixture_crps_matches_gaussian_at_zero_init() -> None:
    # the analytic mixture is M copies of N(y_mean, y_std^2) = N(y_mean, y_std^2):
    # exact equality with crps_gaussian, no MC noise in the score itself.
    from quant_fund.metrics.scoring import crps_gaussian

    y_mean, y_std = -0.4, 0.6
    m = _toy_model(_toy_params(y_mean=y_mean, y_std=y_std))
    ctx = np.zeros((3, _C))
    y = np.array([[0.2, -0.1, 0.9, -0.5]] * 3)
    s_mix = m.mixture_crps(ctx, y, n_samples=64, seed=0)
    s_exact = float(np.mean(crps_gaussian(y.ravel(), np.full(12, y_mean), np.full(12, y_std))))
    np.testing.assert_allclose(s_mix, s_exact, rtol=1e-10)


# ---------------------------------------------------------------------------
# SYNTHETIC simulators / exact oracles (no torch)
# ---------------------------------------------------------------------------


def test_sim_ou_deterministic_and_stationary() -> None:
    a = nsd._sim_ou_paths(64, 12, 0.1, 3.0, 0.0, 0.6, 7)
    b = nsd._sim_ou_paths(64, 12, 0.1, 3.0, 0.0, 0.6, 7)
    np.testing.assert_array_equal(a, b)
    # stationary marginal: N(0, sigma^2 / (2 theta)) = N(0, 0.06).
    np.testing.assert_allclose(a[:, 0].mean(), 0.0, atol=4 * 0.245 / 8)
    assert abs(a[:, 0].var() - 0.06) < 0.06 * 0.6  # 64 draws, loose band


def test_sim_abm_increments_law() -> None:
    x = nsd._sim_abm_paths(256, 8, 0.5, 0.4, 0.5, 3)
    incr = np.diff(x, axis=1)
    np.testing.assert_allclose(
        incr.mean(), 0.4 * 0.5, atol=4 * 0.5 * math.sqrt(0.5) / math.sqrt(256 * 7)
    )
    np.testing.assert_allclose(incr.var(), 0.5 * 0.5 * 0.5, rtol=0.10)


def test_sim_regime_states_and_determinism() -> None:
    x1, s1 = nsd._sim_regime_paths(48, 12, 0.1, (-1.8, 1.8), 0.45, 0.9, 5)
    x2, s2 = nsd._sim_regime_paths(48, 12, 0.1, (-1.8, 1.8), 0.45, 0.9, 5)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(s1, s2)
    assert set(np.unique(s1)) <= {0, 1}


def test_oracle_ou_matches_closed_form() -> None:
    x_last = np.array([0.5, -1.0, 0.0])
    mu, sd = nsd._oracle_ou(x_last, 3, 0.1, 3.0, 0.2, 0.6)
    k = np.arange(1, 4)
    a = np.exp(-3.0 * 0.1 * k)
    np.testing.assert_allclose(mu, 0.2 + (x_last[:, None] - 0.2) * a, rtol=1e-14)
    np.testing.assert_allclose(
        sd,
        np.broadcast_to(np.sqrt(0.6**2 / 6.0 * (1 - a**2)), mu.shape),
        rtol=1e-14,
    )


def test_oracle_abm_matches_closed_form() -> None:
    x_last = np.array([0.5, -1.0])
    mu, sd = nsd._oracle_abm(x_last, 4, 0.25, 0.4, 0.5)
    k = np.arange(1, 5)
    np.testing.assert_allclose(mu, x_last[:, None] + 0.4 * 0.25 * k, rtol=1e-14)
    np.testing.assert_allclose(sd, 0.5 * math.sqrt(0.25) * np.sqrt(k) * np.ones((2, 1)), rtol=1e-14)


def test_oracle_regime_mixture_valid() -> None:
    ctx = nsd._sim_regime_paths(16, 8, 0.125, (-1.5, 1.5), 0.4, 0.85, 2)[0]
    w, mu, sig = nsd._oracle_regime(ctx, 4, 0.125, (-1.5, 1.5), 0.4, 0.85)
    assert w.shape == (16, 4, 16)  # 2^4 regime sequences
    np.testing.assert_allclose(w.sum(axis=-1), 1.0, rtol=1e-12)
    assert np.all(w >= 0.0)
    # unreachable regime sequences carry w == 0 with degenerate components —
    # consumers mask w > 0 (the bench does); positive-weight components need sd.
    assert np.all(sig[w > 0.0] > 0.0)


# ---------------------------------------------------------------------------
# torch-gated: fit / ELBO / inference contract
# ---------------------------------------------------------------------------


@requires_torch
def test_fit_predict_roundtrip_shapes() -> None:
    paths = nsd._sim_ou_paths(48, _T, 1.0 / (_T - 1), 3.0, 0.0, 0.6, 0)
    m = nsd.NeuralSDEForecaster(
        context_len=_C,
        horizon=_H,
        latent_dim=_D,
        sig_order=_SIG,
        ctx_dim=_CTX,
        hidden=(16,),
        epochs=4,
        batch_size=24,
        seed=0,
    ).fit(paths)
    ctx, fut = paths[:6, :_C], paths[:6, _C:]
    assert m.predict_samples(ctx, n_samples=16, seed=0).shape == (6, _H, 16)
    mu, sig = m.predict_components(ctx, n_samples=16, seed=0)
    assert mu.shape == sig.shape == (6, _H, 16)
    assert np.all(sig > 0.0)
    assert m.pit(ctx, fut, n_samples=32, seed=0).shape == (6, _H)
    assert math.isfinite(m.crps(ctx, fut, n_samples=32, seed=0))
    assert math.isfinite(m.mixture_crps(ctx, fut, n_samples=32, seed=0))


@requires_torch
def test_fit_entry0_loss_closed_form() -> None:
    # Engineered init: f=u=0, q0=N(0,I), decoder N(0,1) -> every KL is 0 and
    # loss[0] = T/2 * (mean y_s^2 + log 2pi). Full batch makes the mean exact.
    paths = nsd._sim_ou_paths(32, _T, 1.0 / (_T - 1), 3.0, 0.0, 0.6, 1)
    m = nsd.NeuralSDEForecaster(
        context_len=_C,
        horizon=_H,
        hidden=(16,),
        ctx_dim=_CTX,
        sig_order=_SIG,
        epochs=2,
        batch_size=None,
        seed=0,
    ).fit(paths)
    y_s = (paths - paths.mean()) / paths.std()
    expect = _T * 0.5 * (float(np.mean(y_s**2)) + math.log(2.0 * math.pi))
    fi = m.fit_info
    assert fi is not None
    np.testing.assert_allclose(fi.loss_curve[0], expect, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(fi.kl0_curve[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(fi.kl_path_curve[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(fi.nll_curve[0], expect, rtol=1e-5, atol=1e-5)


@requires_torch
def test_fit_elbo_improves_and_kl_path_turns_positive() -> None:
    paths = nsd._sim_ou_paths(64, _T, 1.0 / (_T - 1), 3.0, 0.0, 0.6, 2)
    m = nsd.NeuralSDEForecaster(
        context_len=_C,
        horizon=_H,
        hidden=(24,),
        ctx_dim=_CTX,
        sig_order=_SIG,
        epochs=12,
        batch_size=32,
        seed=0,
    ).fit(paths)
    fi = m.fit_info
    assert fi is not None
    assert fi.loss_curve[-1] < fi.loss_curve[0]  # ELBO rises
    assert fi.final_kl_path > 0.0  # posterior departs from the prior
    assert fi.final_elbo == pytest.approx(-fi.loss_curve[-1])


@requires_torch
def test_fit_determinism_bit_identical() -> None:
    paths = nsd._sim_ou_paths(48, _T, 1.0 / (_T - 1), 3.0, 0.0, 0.6, 0)
    kw = dict(
        context_len=_C,
        horizon=_H,
        hidden=(16,),
        ctx_dim=_CTX,
        sig_order=_SIG,
        epochs=4,
        batch_size=24,
        seed=7,
    )
    a = nsd.NeuralSDEForecaster(**kw).fit(paths)
    b = nsd.NeuralSDEForecaster(**kw).fit(paths)
    assert a.fit_info is not None and b.fit_info is not None
    assert a.fit_info.loss_curve == b.fit_info.loss_curve
    sa = a.predict_samples(paths[:4, :_C], n_samples=16, seed=1)
    sb = b.predict_samples(paths[:4, :_C], n_samples=16, seed=1)
    np.testing.assert_array_equal(sa, sb)


@requires_torch
def test_fit_input_validation_before_torch() -> None:
    m = nsd.NeuralSDEForecaster(context_len=_C, horizon=_H, epochs=1)
    with pytest.raises(ValueError):  # wrong T
        m.fit(_rng_paths(8, _T + 1))
    with pytest.raises(ValueError):  # too few paths
        m.fit(_rng_paths(4, _T))
    with pytest.raises(ValueError):  # NaN
        bad = _rng_paths(16, _T)
        bad[0, 0] = np.nan
        m.fit(bad)


@requires_torch
def test_fit_constant_paths_fail_closed() -> None:
    # constant paths -> y_std ~ 0 must fail closed.
    m = nsd.NeuralSDEForecaster(context_len=_C, horizon=_H, epochs=1)
    with pytest.raises(ValueError):
        m.fit(np.ones((16, _T)))


@requires_torch
def test_posterior_and_prior_rollouts_differ_after_fit() -> None:
    paths = nsd._sim_regime_paths(64, _T, 1.0 / (_T - 1), (-1.8, 1.8), 0.45, 0.9, 4)[0]
    m = nsd.NeuralSDEForecaster(
        context_len=_C,
        horizon=_H,
        hidden=(24,),
        ctx_dim=_CTX,
        sig_order=_SIG,
        epochs=12,
        batch_size=32,
        seed=0,
    ).fit(paths)
    ctx = paths[:8, :_C]
    sp = m.predict_samples(ctx, n_samples=32, seed=0, drift="posterior")
    sq = m.predict_samples(ctx, n_samples=32, seed=0, drift="prior")
    assert sp.shape == sq.shape
    assert not np.array_equal(sp, sq)  # trained correction moves the draws


@requires_torch
def test_n_ahead_extrapolation_beyond_horizon() -> None:
    paths = _rng_paths(48, _T, seed=9)
    m = nsd.NeuralSDEForecaster(
        context_len=_C,
        horizon=_H,
        hidden=(16,),
        ctx_dim=_CTX,
        sig_order=_SIG,
        epochs=3,
        batch_size=24,
        seed=0,
    ).fit(paths)
    out = m.predict_samples(paths[:3, :_C], n_ahead=6, n_samples=8, seed=0)
    assert out.shape == (3, 6, 8)


# ---------------------------------------------------------------------------
# bench: labels + seeded quality bands (torch-gated)
# ---------------------------------------------------------------------------

_BENCH_KEYS = {
    "synthetic_model_crps",
    "synthetic_model_mixture_crps",
    "synthetic_oracle_crps",
    "synthetic_baseline_crps",
    "synthetic_coverage_50",
    "synthetic_coverage_90",
    "synthetic_width_90",
    "synthetic_oracle_coverage_90",
    "synthetic_oracle_width_90",
    "synthetic_pit_ks",
    "synthetic_pit_ks_pvalue",
    "synthetic_elbo_gain",
    "synthetic_final_kl_path",
    "synthetic_dgp",
    "synthetic_synthetic",
    "synthetic_claim",
}


@requires_torch
def test_bench_ou_labels_and_quality_bands() -> None:
    r = nsd.bench_neural_sde(
        dgp="ou",
        n_train=128,
        n_eval=48,
        epochs=60,
        hidden=(24,),
        ctx_dim=16,
        n_samples=128,
        seed=0,
    )
    assert set(r) >= _BENCH_KEYS
    assert r["synthetic_dgp"] == "ou" and r["synthetic_claim"] == "research_metric_only"
    assert "synthetic" in r["synthetic_synthetic"]
    # seeded SYNTHETIC bands (measured at these settings: model_crps 0.120,
    # oracle 0.104, baseline 0.111, coverage_90 0.917, pit p 0.026):
    # model within 60% of oracle, within 20% of baseline; coverage_90 within
    # ~7 binomial se of nominal; ELBO improves.
    assert r["synthetic_model_crps"] <= 1.6 * r["synthetic_oracle_crps"]
    assert r["synthetic_model_crps"] <= 1.2 * r["synthetic_baseline_crps"]
    assert 0.75 <= r["synthetic_coverage_90"] <= 1.0
    assert 0.75 <= r["synthetic_oracle_coverage_90"] <= 1.0
    assert r["synthetic_pit_ks_pvalue"] > 0.005
    assert r["synthetic_elbo_gain"] > 0.0
    assert r["synthetic_width_90"] > 0.0


@requires_torch
def test_bench_gbm_quality_bands() -> None:
    # on GBM the anchored baseline IS the true law, so the model approaches
    # it from above; bands reflect that (measured: model 0.207 vs baseline
    # 0.135, coverage_90 0.859, pit p 0.093 at 60 epochs).
    r = nsd.bench_neural_sde(
        dgp="gbm",
        n_train=128,
        n_eval=48,
        epochs=60,
        hidden=(24,),
        ctx_dim=16,
        n_samples=128,
        seed=0,
    )
    assert r["synthetic_model_crps"] <= 1.8 * r["synthetic_oracle_crps"]
    assert r["synthetic_model_crps"] <= 1.8 * r["synthetic_baseline_crps"]
    assert 0.70 <= r["synthetic_coverage_90"] <= 1.0
    assert r["synthetic_elbo_gain"] > 0.0


@requires_torch
def test_bench_regime_quality_bands() -> None:
    # regime oracle is the exact HMM-filtered mixture; the model is allowed
    # generous slack at this tiny budget but must stay calibrated in the
    # PIT sense (measured: model 0.456 vs oracle 0.234, coverage_90 0.875,
    # pit p 0.557 at 60 epochs).
    r = nsd.bench_neural_sde(
        dgp="regime",
        n_train=128,
        n_eval=48,
        epochs=60,
        hidden=(24,),
        ctx_dim=16,
        n_samples=128,
        seed=0,
    )
    assert r["synthetic_model_crps"] <= 2.2 * r["synthetic_oracle_crps"]
    assert 0.65 <= r["synthetic_coverage_90"] <= 1.0
    assert r["synthetic_pit_ks_pvalue"] > 0.005
    assert r["synthetic_elbo_gain"] > 0.0


def test_bench_fail_closed_on_bad_dgp() -> None:
    with pytest.raises(ValueError):
        nsd.bench_neural_sde(dgp="heston")
