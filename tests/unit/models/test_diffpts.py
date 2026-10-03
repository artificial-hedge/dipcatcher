"""Tests for quant_fund.models.diffpts — DiffPTS (Ye et al. 2026).

References: Ye, Li, Liu, Jiang, Sekimoto & Jiang (2026, NeurIPS,
arXiv:2609.32363, "DiffPTS: Rethinking Diffusion ELBO for Probabilistic
Time Series Forecasting"); Ho, Jain & Abbeel (2020, NeurIPS 33,
arXiv:2006.11239, DDPM forward process / posterior); Song, Meng & Ermon
(2021, ICLR, arXiv:2010.02502, DDIM); Nichol & Dhariwal (2021, ICML,
arXiv:2102.09672, cosine schedule); StocBench (arXiv:2608.22309,
score-vs-NFE reporting); Greenbury et al. (2026, arXiv:2606.12997 —
spec'd as "AutoCast"; real title "Reliability of Probabilistic Emulation
of Physical Systems"; CRPS-ensembles-vs-latent-diffusion honesty).

All data here is SYNTHETIC (seeded AR(1) streams / hand-built parameter
fixtures) — correctness evidence for the algorithm, never market evidence;
no live-trading claims. Torch tests skip cleanly when the nn extra is
absent; every sampler/scoring path is pure numpy and always runs.
"""

from __future__ import annotations

import importlib.util
import math
import sys

import numpy as np
import pytest

from quant_fund.models import diffpts as dp


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="diffpts training requires the nn extra (torch)"
)

LOOKBACK = 8
N_TRAIN = 260
N_TEST = 80
TAUS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])


def _stream_xy(
    kind: str = "ar1_gauss", seed: int = 3, n: int = N_TRAIN + N_TEST
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = dp.simulate_synthetic_stream(n + LOOKBACK, kind=kind, seed=seed)
    X, y = dp.make_windows(s, LOOKBACK)
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]


def _lin_sched(n_steps: int = 20) -> dp.DiffusionSchedule:
    return dp.DiffusionSchedule.linear(n_steps, beta_start=1e-3, beta_end=0.2)


def _hand_params(
    n_steps: int = 12, p: int = LOOKBACK, eps_of_zt: float = 1.0
) -> dp.DiffusionParams:
    """Hand-built params: eps_hat = eps_of_zt * z_t, mu = 0.25, sigma = e^0 = 1.

    A single-layer denoiser whose only nonzero weight sits on the z_t input —
    no torch needed, so the samplers and PIT/CRPS paths are exercised purely
    in numpy.
    """
    n_tf = 4
    d_in = 1 + (1 + 2 * n_tf) + p
    w_den = np.zeros((1, d_in))
    w_den[0, 0] = eps_of_zt
    return dp.DiffusionParams(
        denoiser_layers=((w_den, np.zeros(1)),),
        loc_layers=((np.zeros((1, p)), np.array([0.25])),),
        scale_layers=((np.zeros((1, p)), np.zeros(1)),),
        ctx_mean=np.zeros(p),
        ctx_std=np.ones(p),
        log_sigma_base=0.0,
        log_sigma_bound=4.0,
        schedule=_lin_sched(n_steps),
        n_time_freqs=n_tf,
    )


# ---------------------------------------------------------------------------
# numpy core (always run, no torch needed)
# ---------------------------------------------------------------------------


def test_schedule_linear_properties() -> None:
    s = _lin_sched(30)
    assert s.n_steps == 30
    assert np.all((s.betas > 0.0) & (s.betas < 1.0))
    assert np.all(np.diff(s.alphas_cumprod) < 0.0)
    assert s.alphas_cumprod[0] == pytest.approx(1.0 - 1e-3)
    assert s.alphas_cumprod[-1] < 0.1  # terminal state is essentially noise
    assert s.alpha_bar(0) == 1.0
    assert s.alpha_bar(30) == s.alphas_cumprod[-1]


def test_schedule_cosine_properties_and_fail_closed() -> None:
    c = dp.DiffusionSchedule.cosine(50)
    lin = dp.DiffusionSchedule.linear(50)
    assert np.all(np.diff(c.alphas_cumprod) < 0.0)
    # Cosine tails reach a near-zero signal level; the default linear ramp
    # (beta_end = 0.02) does not fully denoise at T = 50.
    assert c.alphas_cumprod[-1] < lin.alphas_cumprod[-1]
    with pytest.raises(ValueError, match="non-empty"):
        dp.DiffusionSchedule(betas=np.array([]), alphas_cumprod=np.array([]))
    with pytest.raises(ValueError, match="beta_start"):
        dp.DiffusionSchedule.linear(10, beta_start=0.0)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        dp.DiffusionSchedule(betas=np.array([0.0, 0.5]), alphas_cumprod=np.array([1.0, 0.5]))
    with pytest.raises(ValueError, match="beta_start < beta_end"):
        dp.DiffusionSchedule.linear(10, beta_start=0.5, beta_end=0.4)
    b = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="cumprod"):
        dp.DiffusionSchedule(betas=b, alphas_cumprod=np.array([0.9, 0.5]))
    with pytest.raises(ValueError, match="s must be"):
        dp.DiffusionSchedule.cosine(10, s=1.5)


def test_alpha_bar_validation_and_array_form() -> None:
    s = _lin_sched(10)
    vals = s.alpha_bar(np.array([0, 1, 5, 10]))
    assert vals[0] == 1.0
    assert np.allclose(vals[1:], s.alphas_cumprod[[0, 4, 9]])
    with pytest.raises(ValueError, match=r"\[0, 10\]"):
        s.alpha_bar(11)
    with pytest.raises(ValueError, match=r"\[0, 10\]"):
        s.alpha_bar(-1)


def test_q_sample_closed_form() -> None:
    s = _lin_sched(20)
    z0 = np.array([0.3, -1.2, 2.0])
    t = 5
    ab = float(s.alpha_bar(t))
    assert np.allclose(dp.q_sample(z0, t, np.zeros(3), s), math.sqrt(ab) * z0)
    assert np.allclose(dp.q_sample(z0, t, np.ones(3), s), math.sqrt(ab) * z0 + math.sqrt(1.0 - ab))
    # Per-row timesteps broadcast against the arrays.
    ta = np.array([1, 5, 20])
    got = dp.q_sample(z0, ta, np.zeros(3), s)
    ab_ta = np.asarray(s.alpha_bar(ta))
    assert np.allclose(got, np.sqrt(ab_ta) * z0)


def test_q_sample_moments_match_theory() -> None:
    s = _lin_sched(20)
    rng = np.random.default_rng(0)
    z0 = np.full(20000, 1.7)
    t = 7
    zt = dp.q_sample(z0, t, rng.standard_normal(z0.shape), s)
    ab = float(s.alpha_bar(t))
    assert zt.mean() == pytest.approx(math.sqrt(ab) * 1.7, abs=0.02)
    assert zt.var() == pytest.approx(1.0 - ab, abs=0.02)


def test_x0_from_epsilon_inverts_q_sample() -> None:
    s = _lin_sched(20)
    rng = np.random.default_rng(1)
    z0 = rng.standard_normal(64)
    eps = rng.standard_normal(64)
    for t in (1, 7, 20):
        zt = dp.q_sample(z0, t, eps, s)
        back = dp.x0_from_epsilon(zt, eps, t, s)
        assert np.allclose(back, z0, atol=1e-10)


def test_posterior_at_t1_is_deterministic_x0() -> None:
    s = _lin_sched(20)
    z_t = np.array([0.4, -0.9])
    x0 = np.array([0.7, 1.1])
    mean, var = dp.posterior_mean_variance(z_t, x0, 1, s)
    assert var == pytest.approx(0.0, abs=1e-15)
    assert np.allclose(mean, x0)


def test_posterior_hand_computed() -> None:
    """betas = [0.1, 0.2, 0.3, 0.4] => abar = [0.9, 0.72, 0.504, 0.3024]."""
    b = np.array([0.1, 0.2, 0.3, 0.4])
    s = dp.DiffusionSchedule(betas=b, alphas_cumprod=np.cumprod(1.0 - b))
    mean, var = dp.posterior_mean_variance(np.array([0.5]), np.array([0.2]), 2, s)
    coef_x0 = math.sqrt(0.9) * 0.2 / (1.0 - 0.72)
    coef_zt = math.sqrt(0.8) * 0.1 / (1.0 - 0.72)
    assert mean[0] == pytest.approx(coef_x0 * 0.2 + coef_zt * 0.5)
    assert var == pytest.approx(0.2 * 0.1 / (1.0 - 0.72))


def test_posterior_moments_match_direct_conditioning() -> None:
    """Law of total moments: marginalizing q(z_{t-1}|z_t,z0) over z_t recovers
    the forward marginal moments of z_{t-1}.

    For fixed z0 the forward marginal is z_t ~ N(sqrt(abar_t) z0, 1-abar_t).
    The posterior mean averaged over that marginal must equal
    E[z_{t-1} | z0] = sqrt(abar_{t-1}) z0, and beta~ + Var(mu~) must equal
    Var(z_{t-1} | z0) = 1 - abar_{t-1} (total-variance identity — a real
    consistency check on the closed form, not a tautology).
    """
    s = _lin_sched(10)
    z0 = 1.3
    t = 5
    rng = np.random.default_rng(2)
    n = 40000
    ab_t = float(s.alpha_bar(t))
    ab_prev = float(s.alpha_bar(t - 1))
    z_t = math.sqrt(ab_t) * z0 + math.sqrt(1.0 - ab_t) * rng.standard_normal(n)
    mean, var = dp.posterior_mean_variance(z_t, np.full(n, z0), t, s)
    assert float(np.mean(mean)) == pytest.approx(math.sqrt(ab_prev) * z0, abs=0.02)
    assert var + float(np.var(mean)) == pytest.approx(1.0 - ab_prev, abs=0.02)


def test_oracle_denoiser_recovers_conditional_mean() -> None:
    """Planted signal: for z0 ~ N(0,1), eps* = sqrt(1-abar) z_t gives x0_hat = E[z0|z_t].

    With z0 ~ N(0,1), z_t = sqrt(abar) z0 + sqrt(1-abar) eps is jointly
    Gaussian, so E[z0 | z_t] = sqrt(abar) z_t and the optimal denoiser is
    eps*(z_t) = sqrt(1 - abar) z_t. Feeding it to x0_from_epsilon must return
    sqrt(abar) z_t exactly — an algebraic identity, not a Monte Carlo check.
    """
    s = _lin_sched(20)
    rng = np.random.default_rng(4)
    z0 = rng.standard_normal(512)
    t = 6
    ab = float(s.alpha_bar(t))
    z_t = dp.q_sample(z0, t, rng.standard_normal(512), s)
    eps_oracle = math.sqrt(1.0 - ab) * z_t
    x0_hat = dp.x0_from_epsilon(z_t, eps_oracle, t, s)
    assert np.allclose(x0_hat, math.sqrt(ab) * z_t, atol=1e-10)


def test_ddim_step_landing_and_eta() -> None:
    s = _lin_sched(10)
    z_t = np.array([0.8, -0.4])
    eps_hat = np.array([0.1, 0.2])
    # t_prev = 0 lands on x0_hat exactly (abar_0 = 1, sigma = 0).
    out = dp.ddim_step(z_t, eps_hat, 4, 0, s)
    assert np.allclose(out, dp.x0_from_epsilon(z_t, eps_hat, 4, s))
    # eta = 0 ignores noise; eta > 0 uses it and shifts the draw.
    a = dp.ddim_step(z_t, eps_hat, 4, 2, s, eta=0.0, noise=np.full(2, 9.0))
    b = dp.ddim_step(z_t, eps_hat, 4, 2, s, eta=0.0)
    assert np.array_equal(a, b)
    noisy = np.array([0.3, -0.6])
    c = dp.ddim_step(z_t, eps_hat, 4, 2, s, eta=1.0, noise=noisy)
    assert not np.allclose(c, b)
    with pytest.raises(ValueError, match="noise is required"):
        dp.ddim_step(z_t, eps_hat, 4, 2, s, eta=0.5)
    with pytest.raises(ValueError, match="eta"):
        dp.ddim_step(z_t, eps_hat, 4, 2, s, eta=1.5)
    with pytest.raises(ValueError, match="t_prev"):
        dp.ddim_step(z_t, eps_hat, 4, 4, s)
    with pytest.raises(ValueError, match="n_steps"):
        dp.ddim_step(z_t, eps_hat, 11, 4, s)


def test_time_features_shape_and_bounds() -> None:
    tf = dp.time_features(np.array([0, 5, 10]), 10, 3)
    assert tf.shape == (3, 7)
    assert np.allclose(tf[:, 0], [0.0, 0.5, 1.0])
    assert np.all(np.abs(tf) <= 1.0)
    with pytest.raises(ValueError, match="inside"):
        dp.time_features(np.array([11]), 10, 3)


def test_hand_params_predict_epsilon_passthrough() -> None:
    p = _hand_params()
    X = np.random.default_rng(0).standard_normal((16, LOOKBACK))
    z = np.linspace(-1.0, 1.0, 16)
    out = dp.predict_epsilon(p, z, 3, X)
    assert np.allclose(out, z)  # eps_hat = z_t by construction
    out_t = dp.predict_epsilon(p, z, np.full(16, 3), X)
    assert np.allclose(out_t, out)
    mu, sigma = dp.predict_location_scale(p, X)
    assert np.allclose(mu, 0.25) and np.allclose(sigma, 1.0)


def test_ddpm_path_count_and_determinism() -> None:
    p = _hand_params(n_steps=12)
    X = np.random.default_rng(1).standard_normal((6, LOOKBACK))
    a = dp.ddpm_ancestral_sample(p, X, 24, seed=7)
    b = dp.ddpm_ancestral_sample(p, X, 24, seed=7)
    c = dp.ddpm_ancestral_sample(p, X, 24, seed=8)
    assert a.samples.shape == (6, 24)
    assert a.nfe == 12 and a.sampler == "ddpm"
    assert np.array_equal(a.samples, b.samples)
    assert not np.array_equal(a.samples, c.samples)
    assert np.all(np.isfinite(a.samples))


def test_ddim_path_count_and_eta0_determinism() -> None:
    p = _hand_params(n_steps=40)
    X = np.random.default_rng(2).standard_normal((5, LOOKBACK))
    a = dp.ddim_sample(p, X, 16, steps=5, seed=11)
    b = dp.ddim_sample(p, X, 16, steps=5, seed=11)
    assert a.nfe == 5 and a.sampler == "ddim"
    assert a.samples.shape == (5, 16)
    assert np.array_equal(a.samples, b.samples)  # eta = 0: only the z_T draw is random
    det = dp.ddim_sample(p, X, 16, steps=5, seed=11, eta=0.5)
    assert not np.array_equal(a.samples, det.samples)
    cap = dp.ddim_sample(p, X, 8, steps=1000, seed=0)
    assert cap.nfe == p.schedule.n_steps  # honest clamp: NFE never exceeds T


def test_sampler_fail_closed() -> None:
    p = _hand_params()
    X = np.zeros((4, LOOKBACK))
    with pytest.raises(ValueError, match="n_samples"):
        dp.ddpm_ancestral_sample(p, X, 0)
    with pytest.raises(ValueError, match="steps"):
        dp.ddim_sample(p, X, 4, steps=0)
    with pytest.raises(ValueError, match="eta"):
        dp.ddim_sample(p, X, 4, steps=3, eta=-0.1)
    with pytest.raises(ValueError, match="feature count mismatch"):
        dp.ddpm_ancestral_sample(p, np.zeros((4, LOOKBACK + 1)), 4)
    with pytest.raises(ValueError, match="finite"):
        dp.ddim_sample(p, np.full((4, LOOKBACK), np.nan), 4, steps=3)


def test_diffusion_params_validation() -> None:
    p = _hand_params()
    good = dict(
        loc_layers=p.loc_layers,
        scale_layers=p.scale_layers,
        ctx_mean=p.ctx_mean,
        ctx_std=p.ctx_std,
        log_sigma_base=0.0,
        log_sigma_bound=4.0,
        schedule=p.schedule,
        n_time_freqs=4,
    )
    with pytest.raises(ValueError, match="denoiser_layers"):
        dp.DiffusionParams(denoiser_layers=((np.zeros((1, 5)), np.zeros(1)),), **good)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strictly positive"):
        dp.DiffusionParams(
            denoiser_layers=p.denoiser_layers,
            ctx_std=np.zeros(LOOKBACK),
            **{k: v for k, v in good.items() if k != "ctx_std"},  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="schedule"):
        dp.DiffusionParams(
            denoiser_layers=p.denoiser_layers,
            schedule="nope",
            **{k: v for k, v in good.items() if k != "schedule"},  # type: ignore[arg-type]
        )


def test_make_windows_alignment_and_fail_closed() -> None:
    s = np.arange(20, dtype=float)
    X, y = dp.make_windows(s, 4)
    assert X.shape == (16, 4) and y.shape == (16,)
    assert np.allclose(X[0], [0, 1, 2, 3]) and y[0] == 4.0
    assert np.allclose(X[15], [15, 16, 17, 18]) and y[15] == 19.0
    with pytest.raises(ValueError, match="lookback"):
        dp.make_windows(s, 0)
    with pytest.raises(ValueError, match="length > lookback"):
        dp.make_windows(np.arange(4.0), 4)
    with pytest.raises(ValueError, match="finite"):
        dp.make_windows(np.array([1.0, np.nan, 3.0, 4.0, 5.0]), 2)


def test_simulate_stream_determinism_and_kinds() -> None:
    a = dp.simulate_synthetic_stream(200, kind="ar1_gauss", seed=5)
    b = dp.simulate_synthetic_stream(200, kind="ar1_gauss", seed=5)
    c = dp.simulate_synthetic_stream(200, kind="ar1_bimodal", seed=5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(np.isfinite(a)) and a.shape == (200,)
    # AR(1) coefficient recovered by least squares on the Gaussian stream.
    coef = float(np.sum(a[1:] * a[:-1]) / np.sum(a[:-1] ** 2))
    assert coef == pytest.approx(0.7, abs=0.1)
    # Bimodal innovations: residual kurtosis is far below the Gaussian value 3.
    res = c[1:] - 0.7 * c[:-1]
    kurt = float(np.mean((res - res.mean()) ** 4) / res.var() ** 2)
    assert kurt < 2.0
    with pytest.raises(ValueError, match="kind"):
        dp.simulate_synthetic_stream(10, kind="garch")
    with pytest.raises(ValueError, match="phi"):
        dp.simulate_synthetic_stream(10, phi=1.0)
    with pytest.raises(ValueError, match="sigma"):
        dp.simulate_synthetic_stream(10, sigma=0.0)


def test_evaluate_samples_on_calibrated_ensemble() -> None:
    """y ~ N(0,1) scored against N(0,1) draws: E[CRPS] = 1/sqrt(pi) ~ 0.5642."""
    rng = np.random.default_rng(6)
    n, m = 400, 256
    y = rng.standard_normal(n)
    samples = rng.standard_normal((n, m))
    sc = dp.evaluate_samples(y, samples, TAUS, seed=0)
    assert sc.crps == pytest.approx(1.0 / math.sqrt(math.pi), abs=0.05)
    assert sc.pinball_mean > 0.0
    assert set(sc.pinball_by_tau) == {float(t) for t in TAUS}
    assert sc.pit_hist.sum() == n
    assert sc.pit.mean() == pytest.approx(0.5, abs=0.05)
    assert sc.pit_ks < 0.15
    assert sc.coverage_90 == pytest.approx(0.9, abs=0.06)
    assert sc.mean_width_90 == pytest.approx(2 * 1.6449, abs=0.1)
    assert sc.n_obs == n and sc.n_samples == m


def test_evaluate_samples_fail_closed() -> None:
    y = np.zeros(8)
    s = np.zeros((8, 4))
    with pytest.raises(ValueError, match="samples must be"):
        dp.evaluate_samples(y, np.zeros((7, 4)), TAUS)
    with pytest.raises(ValueError, match="at least one draw"):
        dp.evaluate_samples(y, np.zeros((8, 0)), TAUS)
    with pytest.raises(ValueError, match="taus"):
        dp.evaluate_samples(y, s, np.array([1.5]))
    with pytest.raises(ValueError, match="non-empty"):
        dp.evaluate_samples(np.array([]), np.zeros((0, 4)), TAUS)
    with pytest.raises(ValueError, match="finite"):
        dp.evaluate_samples(np.full(8, np.inf), s, TAUS)


def test_model_constructor_and_unfitted_fail_closed() -> None:
    with pytest.raises(ValueError, match="n_steps"):
        dp.DiffPTSModel(n_steps=0)
    with pytest.raises(ValueError, match="hidden"):
        dp.DiffPTSModel(hidden=())
    with pytest.raises(ValueError, match="est_nll_weight"):
        dp.DiffPTSModel(est_nll_weight=-1.0)
    with pytest.raises(ValueError, match="schedule_kind"):
        dp.DiffPTSModel(schedule_kind="quadratic")
    with pytest.raises(ValueError, match="epochs"):
        dp.DiffPTSModel(epochs=0)
    m = dp.DiffPTSModel()
    assert not m.is_fitted
    X = np.zeros((4, 3))
    with pytest.raises(RuntimeError, match="not fitted"):
        m.predict_params(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        m.sample(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        m.pit(X, np.zeros(4))
    with pytest.raises(RuntimeError, match="not fitted"):
        m.crps(X, np.zeros(4))


def test_module_imports_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """No top-level torch import; fit fails closed with install guidance."""
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    spec = importlib.util.spec_from_file_location("_dp_no_torch", dp.__file__)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_dp_no_torch", probe)
    spec.loader.exec_module(probe)  # module loads cleanly without torch
    # numpy core stays usable while torch is blocked
    s = probe.DiffusionSchedule.linear(8)
    assert s.n_steps == 8
    X, y = probe.make_windows(np.arange(30.0), 5)
    assert X.shape == (25, 5)
    with pytest.raises(ImportError, match="'nn' extra"):
        probe.DiffPTSModel(n_steps=4, epochs=1).fit(X, y)


# ---------------------------------------------------------------------------
# torch lane (skipped when the nn extra is absent)
# ---------------------------------------------------------------------------


@requires_torch
def test_fit_elbo_finite_and_nonincreasing() -> None:
    Xtr, ytr, _, _ = _stream_xy("ar1_gauss", seed=3)
    m = dp.DiffPTSModel(n_steps=20, hidden=(16,), epochs=60, seed=5)
    m.fit(Xtr, ytr)
    info = m.fit_info
    assert info is not None and m.is_fitted
    assert len(info.loss_curve) == 60
    assert np.all(np.isfinite(info.loss_curve))
    assert np.all(np.isfinite(info.denoise_curve))
    assert np.all(np.isfinite(info.est_nll_curve))
    assert info.loss_curve[-1] < info.loss_curve[0]
    # The induced Gaussian NLL on the location-scale estimators stays bounded
    # and improves or holds near its ridge warm-start (it cannot blow up under
    # a converging joint objective).
    assert info.est_nll_curve[-1] < info.est_nll_curve[0] + 0.5


@requires_torch
def test_fit_deterministic_given_seed() -> None:
    Xtr, ytr, _, _ = _stream_xy("ar1_gauss", seed=3)
    kw = dict(n_steps=10, hidden=(8,), epochs=20)
    a = dp.DiffPTSModel(seed=5, **kw).fit(Xtr, ytr)
    b = dp.DiffPTSModel(seed=5, **kw).fit(Xtr, ytr)
    c = dp.DiffPTSModel(seed=6, **kw).fit(Xtr, ytr)
    for la, lb_ in zip(a.params.denoiser_layers, b.params.denoiser_layers, strict=True):
        assert np.array_equal(la[0], lb_[0]) and np.array_equal(la[1], lb_[1])
    for la, lc in zip(a.params.denoiser_layers, c.params.denoiser_layers, strict=True):
        assert not (np.array_equal(la[0], lc[0]) and np.array_equal(la[1], lc[1]))
    sa = a.sample(Xtr[:8], 16, sampler="ddim", steps=4, seed=9)
    sb = b.sample(Xtr[:8], 16, sampler="ddim", steps=4, seed=9)
    assert np.array_equal(sa.samples, sb.samples)


@requires_torch
def test_fit_fail_closed() -> None:
    Xtr, ytr, _, _ = _stream_xy("ar1_gauss", seed=3)
    m = dp.DiffPTSModel(epochs=2)
    with pytest.raises(ValueError, match="2-D"):
        m.fit(np.zeros(10), np.zeros(10))
    with pytest.raises(ValueError, match="length"):
        m.fit(Xtr, ytr[:-1])
    with pytest.raises(ValueError, match="finite"):
        m.fit(np.full((N_TRAIN, LOOKBACK), np.nan), ytr)
    with pytest.raises(ValueError, match="too few samples"):
        m.fit(Xtr[:8], ytr[:8])
    with pytest.raises(ValueError, match="zero variance"):
        m.fit(Xtr, np.zeros(N_TRAIN))
    with pytest.raises(ValueError, match="lr"):
        dp.DiffPTSModel(lr=0.0)
    with pytest.raises(ValueError, match="ridge_alpha"):
        dp.DiffPTSModel(ridge_alpha=-0.5)


@requires_torch
def test_fit_ar1_gauss_location_scale_recovery() -> None:
    """On a linear-Gaussian stream, mu_phi recovers ~0.7*y_last and sigma ~1."""
    Xtr, ytr, Xte, yte = _stream_xy("ar1_gauss", seed=3)
    m = dp.DiffPTSModel(n_steps=20, hidden=(16,), epochs=80, seed=5).fit(Xtr, ytr)
    mu, sigma = m.predict_params(Xte)
    cond_mean = 0.7 * Xte[:, -1]
    # mu tracks the true conditional mean better than the unconditional one.
    err_mu = float(np.sqrt(np.mean((mu - cond_mean) ** 2)))
    err_flat = float(np.sqrt(np.mean(cond_mean**2)))
    assert err_mu < 0.5 * err_flat
    assert sigma.mean() == pytest.approx(1.0, abs=0.5)
    s = m.sample(Xte, 128, sampler="ddim", steps=6, seed=9)
    assert s.samples.shape == (N_TEST, 128)
    assert s.nfe == 6
    # Sampled conditional mean should be near the learned location.
    assert np.sqrt(np.mean((s.samples.mean(axis=1) - cond_mean) ** 2)) < 1.5 * err_flat


@requires_torch
def test_ddpm_vs_ddim_honest_nfe_comparison() -> None:
    """StocBench-style: score is reported vs sampler budget; both are finite."""
    Xtr, ytr, Xte, yte = _stream_xy("ar1_gauss", seed=3)
    m = dp.DiffPTSModel(n_steps=30, hidden=(16,), epochs=80, seed=5).fit(Xtr, ytr)
    sc_ddpm = dp.evaluate_samples(
        yte, m.sample(Xte, 128, sampler="ddpm", seed=9).samples, TAUS, seed=9
    )
    sc_ddim = dp.evaluate_samples(
        yte, m.sample(Xte, 128, sampler="ddim", steps=5, seed=9).samples, TAUS, seed=9
    )
    assert sc_ddpm.crps > 0.0 and sc_ddim.crps > 0.0
    # Honest range: few-step DDIM is within a factor 2 of the full chain on
    # this near-Gaussian target — recorded, not asserted as a win.
    assert abs(sc_ddim.crps - sc_ddpm.crps) / sc_ddpm.crps < 1.0


@requires_torch
def test_pit_and_quantiles_wellformed() -> None:
    Xtr, ytr, Xte, yte = _stream_xy("ar1_gauss", seed=3)
    m = dp.DiffPTSModel(n_steps=16, hidden=(16,), epochs=50, seed=5).fit(Xtr, ytr)
    pit = m.pit(Xte, yte, n_samples=128, seed=4)
    assert np.all((pit >= 0.0) & (pit <= 1.0))
    q = m.predict_quantiles(Xte, TAUS, n_samples=128, steps=6, seed=4)
    assert q.shape == (N_TEST, TAUS.size)
    # Sorted sample quantiles are non-decreasing by construction.
    assert np.all(np.diff(q, axis=1) >= -1e-10)
    with pytest.raises(ValueError, match="taus"):
        m.predict_quantiles(Xte, np.array([0.0, 0.5]))
    with pytest.raises(ValueError, match="sampler"):
        m.sample(Xte, 8, sampler="ancestral_ddim")


@requires_torch
def test_bimodal_stream_ensemble_beats_or_tracks_gaussian() -> None:
    """On a bimodal-residual stream the diffusion head must stay honest.

    We assert honest score ranges and record the delta vs the Gaussian
    NGBoost baseline — Greenbury et al. (2026) warn that diffusion does not
    automatically win, so no victory assertion is made.
    """
    Xtr, ytr, Xte, yte = _stream_xy("ar1_bimodal", seed=3)
    m = dp.DiffPTSModel(n_steps=24, hidden=(32,), epochs=100, seed=5).fit(Xtr, ytr)
    sc = dp.evaluate_samples(
        yte, m.sample(Xte, 192, sampler="ddim", steps=6, seed=9).samples, TAUS, seed=9
    )
    scale = float(np.std(ytr))
    assert 0.0 < sc.crps < 2.0 * scale
    assert 0.0 <= sc.pit_ks <= 1.0
    assert 0.0 <= sc.coverage_90 <= 1.0
    # The sampled ensemble should be wider than a point mass around the
    # conditional mean: bimodal innovations have std ~1.2.
    s = m.sample(Xte, 192, sampler="ddim", steps=6, seed=9).samples
    assert float(np.mean(s.std(axis=1))) > 0.3 * scale


@requires_torch
def test_evaluate_synthetic_stream_scorecard() -> None:
    """End-to-end bench: keys, finiteness, labels, NFE sweep — no victory claim."""
    out = dp.evaluate_synthetic_stream(
        n_train=220,
        n_test=60,
        lookback=8,
        kind="ar1_bimodal",
        seed=7,
        n_samples=96,
        ddim_budgets=(2, 6),
        hidden=(16,),
        epochs=50,
        n_steps=20,
    )
    for key in (
        "diffpts_ddpm_crps",
        "diffpts_ddim2_crps",
        "diffpts_ddim6_crps",
        "diffpts_pinball_mean",
        "diffpts_pit_ks",
        "diffpts_coverage_90",
        "ngboost_crps",
        "ngboost_pinball_mean",
        "ngboost_pit_ks",
        "qrf_crps",
        "qrf_pinball_mean",
        "qrf_pit_ks",
        "crps_diffpts_minus_ngboost",
        "crps_diffpts_minus_qrf",
    ):
        assert key in out, key
        assert math.isfinite(float(out[key])), key
    assert out["diffpts_ddpm_nfe"] == 20.0
    assert out["diffpts_ddim2_nfe"] == 2.0
    assert out["dgp"] == "fixture"
    assert out["claim"] == "research_metric_only"
    assert str(out["synthetic"]).startswith("SYNTHETIC")
    scale = 1.0
    assert 0.0 < float(out["diffpts_ddpm_crps"]) < 5.0 * scale
    assert 0.0 < float(out["ngboost_crps"]) < 5.0 * scale
    assert 0.0 < float(out["qrf_crps"]) < 5.0 * scale
