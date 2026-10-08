"""Tests for quant_fund.models.deep_regime_mixture — DeRegiME (Wood et al. 2026).

References: Wood, K., Zohren, S. & Roberts, S.J. (2026), "DeRegiME: Deep
Regime Mixtures for Probabilistic Forecasting under Distribution Shift",
arXiv:2605.19231 (verified against the arXiv abs page and HTML v1, submitted
19 May 2026): stick-breaking gate (their Eq. 5), regime-mixing kernel and
PSD Theorem 2 / direct-sum Proposition 1 (their §3), Student-t mixture
predictive density with Gauss-Hermite marginalisation of the residual
(their Eq. 3, Q=20 nodes), R_eff average-gate-mass threshold 1e-2 (their
§3), propriety of the predictive density (their §4 / Appendix B.2), and the
NLPD headline (20.3% over a DeepAR/GluonTS-style dynamic Student-t head on
ten real benchmarks — NOT reproduced here; the NGBoostGaussian comparison is
a documented SYNTHETIC lane adaptation). Also: Sethuraman (1994) and
Ishwaran & James (2001) for stick-breaking; Duan et al. (2020,
arXiv:1910.03225) for the NGBoost baseline; Hamilton (1989) for the
regime-switch generator.

All data here is SYNTHETIC (simulated regime-switch heteroskedastic
streams) — correctness evidence for the mechanism, never market evidence;
no live-trading claims. Scores are proper (NLPD/log score) only. Torch
tests skip cleanly when the nn extra is absent; the numpy core (gate math,
kernel, mixture density, quadrature) always runs.
"""

from __future__ import annotations

import importlib.util
import math
import sys

import numpy as np
import pytest
from scipy.stats import t as student_t_dist

from quant_fund.models import deep_regime_mixture as drm


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="DeRegiME fitting requires the nn extra (torch)"
)

PLANTED_SIGMAS = (0.3, 0.8, 1.6)  # 3 planted volatility regimes
# Documented Monte-Carlo tolerance for the SYNTHETIC NLPD comparison: with
# the fixed protocol (seed=0 train stream, eval_seed=101 scoring stream) the
# observed gap is about -0.33 nats (DeRegiME better); across train seeds
# {0,1,2} the observed gaps stayed negative (about -0.34 to -0.06). The
# assertion allows +0.05 nats of headroom for platform float differences.
NLPD_TOLERANCE = 0.05
SCALE_RECOVERY_TOL = 0.35


# ---------------------------------------------------------------------------
# numpy core: gate math (always run, no torch needed)
# ---------------------------------------------------------------------------


def test_stick_break_weights_hand_case_and_simplex() -> None:
    # gamma = 0 -> v = 1/2 -> pi = (1/2, 1/2) exactly (paper Eq. 5).
    assert np.allclose(drm.stick_break_weights(np.array([0.0])), [0.5, 0.5])
    logits = np.array([[-1.0, 0.5, 2.0], [0.0, 0.0, 0.0]])
    pi = drm.stick_break_weights(logits)
    assert pi.shape == (2, 4)
    assert np.allclose(pi.sum(axis=-1), 1.0, atol=1e-12)
    assert np.all(pi > 0.0) and np.all(pi < 1.0)  # open simplex, finite logits
    # manual recursion for the first row: v = sigmoid(gamma)
    v = 1.0 / (1.0 + np.exp(-logits[0]))
    manual = np.array(
        [
            v[0],
            v[1] * (1 - v[0]),
            v[2] * (1 - v[0]) * (1 - v[1]),
            (1 - v[0]) * (1 - v[1]) * (1 - v[2]),
        ]
    )
    assert np.allclose(pi[0], manual, atol=1e-12)


def test_stick_break_weights_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite"):
        drm.stick_break_weights(np.array([np.nan]))
    with pytest.raises(ValueError, match="last axis"):
        drm.stick_break_weights(np.zeros((2, 0)))


def test_gate_weights_softmax_and_unknown_gate() -> None:
    logits = np.array([[1.0, 2.0, 3.0]])
    p = drm.gate_weights(logits, "softmax")
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert p.shape == (1, 3)
    # uniform logits -> uniform weights
    assert np.allclose(drm.gate_weights(np.zeros((4, 3)), "softmax"), 1.0 / 3.0)
    with pytest.raises(ValueError, match="unknown gate"):
        drm.gate_weights(logits, "sparsemax")


def test_effective_regime_count_thresholding() -> None:
    # two regimes with average mass above 1e-2, three below (paper §3 R_eff)
    gates = np.zeros((100, 5))
    gates[:, 0] = 0.6
    gates[:, 1] = 0.39
    gates[:, 2] = 0.005
    gates[:, 3] = 0.004
    gates[:, 4] = 0.001
    assert drm.effective_regime_count(gates) == 2
    # uniform gate over 4 regimes -> all effective
    assert drm.effective_regime_count(np.full((10, 4), 0.25)) == 4
    assert drm.effective_regime_count(gates, threshold=0.5) == 1


def test_effective_regime_count_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least 2-D"):
        drm.effective_regime_count(np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="non-negative"):
        drm.effective_regime_count(np.array([[-0.5, 1.5]]))
    with pytest.raises(ValueError, match="threshold"):
        drm.effective_regime_count(np.full((4, 2), 0.5), threshold=0.0)


# ---------------------------------------------------------------------------
# numpy core: regime-mixing kernel (paper Eq. 2, Proposition 1, Theorem 2)
# ---------------------------------------------------------------------------


def test_regime_mix_kernel_is_psd_for_arbitrary_real_gates() -> None:
    # Theorem 2: PSD for ANY real-valued gate function — including negative
    # gates, which are valid for the kernel but not for the density.
    rng = np.random.default_rng(7)
    features = rng.normal(size=(3, 12, 4))
    gates = rng.normal(size=(12, 3))
    k = drm.regime_mix_kernel(
        features,
        gates,
        amplitudes=np.array([1.0, 0.5, 2.0]),
        lengthscales=np.array([1.0, 0.7, 1.3]),
    )
    assert k.shape == (12, 12)
    assert np.allclose(k, k.T, atol=1e-12)
    assert np.linalg.eigvalsh(k).min() >= -1e-9


def test_regime_mix_kernel_disjoint_gates_decorrelate() -> None:
    # Proposition 1 consequence: locations with disjoint-regime gates are
    # decorrelated under the GP prior.
    rng = np.random.default_rng(3)
    features = rng.normal(size=(2, 8, 4))
    gates = np.zeros((8, 2))
    gates[:4, 0] = 1.0  # first block lives in regime 0
    gates[4:, 1] = 1.0  # second block lives in regime 1
    k = drm.regime_mix_kernel(features, gates, amplitudes=np.ones(2), lengthscales=np.ones(2))
    assert np.allclose(k[:4, 4:], 0.0, atol=1e-12)
    assert np.allclose(k[4:, :4], 0.0, atol=1e-12)
    assert k[:4, :4].sum() > 0.0 and k[4:, 4:].sum() > 0.0


def test_regime_mix_kernel_fail_closed() -> None:
    feats = np.zeros((2, 5, 3))
    gates = np.zeros((5, 2))
    ok = {"amplitudes": np.ones(2), "lengthscales": np.ones(2)}
    with pytest.raises(ValueError, match=r"\(R, n, d\)"):
        drm.regime_mix_kernel(np.zeros((5, 3)), gates, **ok)
    with pytest.raises(ValueError, match="gates must be"):
        drm.regime_mix_kernel(feats, np.zeros((5, 3)), **ok)
    with pytest.raises(ValueError, match="strictly positive"):
        drm.regime_mix_kernel(
            feats, gates, amplitudes=np.array([1.0, 0.0]), lengthscales=np.ones(2)
        )
    with pytest.raises(ValueError, match="strictly positive"):
        drm.regime_mix_kernel(
            feats, gates, amplitudes=np.ones(2), lengthscales=np.array([1.0, -1.0])
        )
    with pytest.raises(ValueError, match="finite"):
        drm.regime_mix_kernel(np.full((2, 5, 3), np.nan), gates, **ok)


# ---------------------------------------------------------------------------
# numpy core: predictive density, propriety, and proper scoring
# ---------------------------------------------------------------------------


def test_student_t_logpdf_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    mu = rng.normal(size=50) * 0.2
    sigma = np.exp(rng.normal(size=50) * 0.3)
    nu = rng.uniform(2.5, 20.0)
    expected = student_t_dist.logpdf(y, df=nu, loc=mu, scale=sigma)
    assert np.allclose(drm.student_t_logpdf(y, mu, sigma, nu), expected, atol=1e-12)
    with pytest.raises(ValueError, match="strictly positive"):
        drm.student_t_logpdf(y, mu, -sigma, nu)


@pytest.mark.parametrize("family", ["student_t", "gaussian"])
def test_predictive_density_integrates_to_one(family: str) -> None:
    # Paper §4 / Appendix B.2 "Proper density": with simplex gates, proper
    # components and a proper Gaussian q(delta), the predictive density
    # integrates to one in y (Tonelli). Verified by trapezoid quadrature.
    nu = np.array([4.0, 6.0]) if family == "student_t" else np.ones(2)
    scales = np.array([[[1.0, 2.0]]])  # (1, 1, R=2)
    gates = np.array([[[0.35, 0.65]]])
    half = 150.0 if family == "student_t" else 40.0  # polynomial vs Gaussian tails
    grid = np.linspace(-half, half, 150_001)[:, None]  # (m, 1) broadcast over H=1
    mu = np.zeros((1, 1))
    dmean = np.full((1, 1), 0.4)
    dvar = np.full((1, 1), 0.25)
    dens = np.exp(
        drm.mixture_log_density(
            grid,
            np.broadcast_to(mu, grid.shape),
            delta_mean=np.broadcast_to(dmean, grid.shape),
            delta_var=np.broadcast_to(dvar, grid.shape),
            scales=np.broadcast_to(scales, (grid.shape[0], 1, 2)),
            tails=nu,
            gates=np.broadcast_to(gates, (grid.shape[0], 1, 2)),
            family=family,
        )
    )
    # far-tail densities underflow to exactly 0.0 (float64) — that is fine
    assert np.all(np.isfinite(dens)) and np.all(dens >= 0.0) and dens.max() > 0.0
    assert abs(float(np.trapezoid(dens[:, 0], grid[:, 0])) - 1.0) < 1e-5


def test_nlpd_of_true_generator_dominates_perturbed_variants() -> None:
    # Propriety invariant (proper scoring rule): on data drawn from the true
    # generative mixture, the true density's NLPD is at most that of
    # perturbed variants (shifted mean, inflated/deflated scales, uniform
    # gate). Seeded SYNTHETIC sample.
    rng = np.random.default_rng(11)
    n, n_h, n_r = 4000, 2, 2
    nu = np.array([5.0, 8.0])
    scales = np.empty((n, n_h, n_r))
    scales[..., 0] = 0.7
    scales[..., 1] = 1.8
    gates = np.empty((n, n_h, n_r))
    gates[..., 0] = 0.4
    gates[..., 1] = 0.6
    mu = rng.normal(size=(n, n_h)) * 0.1
    dmean = np.zeros((n, n_h))
    dvar = np.full((n, n_h), 0.04)
    # draw from the true generator: regime ~ pi, delta ~ q, y ~ StudentT_r
    comps = rng.choice(n_r, size=(n, n_h), p=[0.4, 0.6])
    delta = rng.normal(size=(n, n_h)) * math.sqrt(0.04)
    draw = rng.standard_t(nu[comps]) * np.where(comps == 0, 0.7, 1.8)
    y = mu + delta + draw

    def nlpd(m: np.ndarray, s: np.ndarray, g: np.ndarray) -> float:
        return drm.mixture_nlpd(y, m, delta_mean=dmean, delta_var=dvar, scales=s, tails=nu, gates=g)

    true_nlpd = nlpd(mu, scales, gates)
    uniform_gates = np.full_like(gates, 0.5)
    variants = {
        "mean_shift": nlpd(mu + 0.6, scales, gates),
        "scale_inflated": nlpd(mu, scales * 1.6, gates),
        "scale_deflated": nlpd(mu, scales * 0.6, gates),
        "gate_uniform": nlpd(mu, scales, uniform_gates),
    }
    for name, value in variants.items():
        assert true_nlpd <= value + 1e-9, f"true generator lost to {name}: {true_nlpd} vs {value}"
    # strict dominance for the mean shift at this sample size
    assert true_nlpd < variants["mean_shift"] - 0.01


def test_mixture_density_fail_closed() -> None:
    y = np.zeros((2, 1))
    kw = {
        "delta_mean": np.zeros((2, 1)),
        "delta_var": np.ones((2, 1)),
        "scales": np.ones((2, 1, 2)),
        "tails": np.array([4.0, 4.0]),
        "gates": np.full((2, 1, 2), 0.5),
    }
    with pytest.raises(ValueError, match="2-D"):
        drm.mixture_log_density(np.zeros(2), np.zeros((2, 1)), **kw)
    with pytest.raises(ValueError, match=r"scales must be"):
        drm.mixture_log_density(y, np.zeros((2, 1)), **{**kw, "scales": np.ones((2, 2, 2))})
    with pytest.raises(ValueError, match="simplex"):
        drm.mixture_log_density(y, np.zeros((2, 1)), **{**kw, "gates": np.full((2, 1, 2), 0.4)})
    with pytest.raises(ValueError, match="non-negative"):
        drm.mixture_log_density(y, np.zeros((2, 1)), **{**kw, "delta_var": -np.ones((2, 1))})
    with pytest.raises(ValueError, match="strictly positive"):
        drm.mixture_log_density(y, np.zeros((2, 1)), **{**kw, "scales": -np.ones((2, 1, 2))})
    with pytest.raises(ValueError, match="unknown family"):
        drm.mixture_log_density(y, np.zeros((2, 1)), family="laplace", **kw)
    with pytest.raises(ValueError, match="n_nodes"):
        drm.mixture_log_density(y, np.zeros((2, 1)), n_nodes=2, **kw)


# ---------------------------------------------------------------------------
# numpy core: SYNTHETIC regime-switch stream
# ---------------------------------------------------------------------------


def test_simulate_regime_stream_shapes_determinism_and_causality() -> None:
    a = drm.simulate_regime_stream(800, sigmas=PLANTED_SIGMAS, seed=5)
    b = drm.simulate_regime_stream(800, sigmas=PLANTED_SIGMAS, seed=5)
    c = drm.simulate_regime_stream(800, sigmas=PLANTED_SIGMAS, seed=6)
    assert a.y.shape == (800,) and a.regimes.shape == (800,)
    assert a.X.shape[1] == 7  # window=5 lags + rolling std + mean |y|
    assert a.Y.shape == (a.X.shape[0], 2) and a.target_regimes.shape == a.Y.shape
    assert np.array_equal(a.y, b.y) and np.array_equal(a.X, b.X) and np.array_equal(a.Y, b.Y)
    assert not np.array_equal(a.y, c.y)
    # features are causal: row i rebuilds exactly from y history at t = i + 4
    i = 10
    t = i + 4
    hist = a.y[t - 4 : t + 1]
    assert np.allclose(a.X[i, :5], hist[::-1])
    assert np.isclose(a.X[i, 5], float(np.std(hist)))
    assert np.isclose(a.X[i, 6], float(np.mean(np.abs(hist))))
    # targets are the future values, with their planted regimes
    assert np.allclose(a.Y[i, 0], a.y[t + 1]) and np.allclose(a.Y[i, 1], a.y[t + 2])
    assert a.target_regimes[i, 0] == a.regimes[t + 1]


def test_simulate_regime_stream_persistence_and_scale_recovery() -> None:
    s = drm.simulate_regime_stream(
        40_000, sigmas=PLANTED_SIGMAS, p_stay=0.99, tail_nu=None, seed=17
    )
    stay = np.mean(s.regimes[1:] == s.regimes[:-1])
    assert abs(stay - 0.99) < 0.01
    for r, sig in enumerate(PLANTED_SIGMAS):
        realized = float(np.std(s.y[s.regimes == r]))
        assert abs(realized - sig) / sig < 0.1  # generator sanity (Gaussian noise)
    assert set(np.unique(s.regimes)) == {0, 1, 2}


def test_simulate_regime_stream_fail_closed() -> None:
    with pytest.raises(ValueError, match="sigmas"):
        drm.simulate_regime_stream(500, sigmas=(1.0,))
    with pytest.raises(ValueError, match="sigmas"):
        drm.simulate_regime_stream(500, sigmas=(1.0, -2.0))
    with pytest.raises(ValueError, match="n_steps"):
        drm.simulate_regime_stream(8, sigmas=PLANTED_SIGMAS)
    with pytest.raises(ValueError, match="p_stay"):
        drm.simulate_regime_stream(500, sigmas=PLANTED_SIGMAS, p_stay=1.0)
    with pytest.raises(ValueError, match="tail_nu"):
        drm.simulate_regime_stream(500, sigmas=PLANTED_SIGMAS, tail_nu=1.5)


# ---------------------------------------------------------------------------
# torch gating: module imports without torch; torch entry points fail closed
# ---------------------------------------------------------------------------


def test_module_imports_without_torch_and_raises_clear_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module has no top-level torch import; torch entry points fail closed."""
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    spec = importlib.util.spec_from_file_location("_drm_no_torch", drm.__file__)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_drm_no_torch", probe)
    spec.loader.exec_module(probe)  # must import cleanly without torch
    # numpy core stays usable while torch is blocked
    assert np.allclose(probe.stick_break_weights(np.array([0.0])), [0.5, 0.5])
    stream = probe.simulate_regime_stream(400, sigmas=PLANTED_SIGMAS, seed=0)
    assert stream.X.shape[0] > 0
    with pytest.raises(ImportError, match=r"'nn' extra"):
        probe.fit_deregime(stream.X, stream.Y, epochs=1)
    with pytest.raises(ImportError, match=r"'nn' extra"):
        probe.synthetic_regime_benchmark(epochs=1)


# ---------------------------------------------------------------------------
# torch lane (skipped when the nn extra is absent)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_fit() -> tuple[drm.SyntheticRegimeStream, drm.DeRegiMEResult]:
    """Small seeded SYNTHETIC two-regime stream + quick fit for shape checks."""
    stream = drm.simulate_regime_stream(
        700, sigmas=(0.5, 1.5), n_horizons=2, p_stay=0.98, tail_nu=None, seed=7
    )
    res = drm.fit_deregime(stream.X, stream.Y, n_regimes=3, epochs=150, sigma_floor=0.1, seed=0)
    return stream, res


@requires_torch
def test_fit_fail_closed() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    y = rng.normal(size=(40, 2))
    with pytest.raises(ValueError, match="2-D"):
        drm.fit_deregime(x, y[:, 0])
    with pytest.raises(ValueError, match="row mismatch"):
        drm.fit_deregime(x, y[:39])
    with pytest.raises(ValueError, match="finite"):
        drm.fit_deregime(x, np.full_like(y, np.nan))
    with pytest.raises(ValueError, match="n_regimes"):
        drm.fit_deregime(x, y, n_regimes=1)
    with pytest.raises(ValueError, match="unknown gate"):
        drm.fit_deregime(x, y, gate="top_k")
    with pytest.raises(ValueError, match="unknown family"):
        drm.fit_deregime(x, y, family="laplace")
    with pytest.raises(ValueError, match="epochs"):
        drm.fit_deregime(x, y, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        drm.fit_deregime(x, y, lr=0.0)
    with pytest.raises(ValueError, match="hidden"):
        drm.fit_deregime(x, y, hidden=())
    with pytest.raises(ValueError, match="gate_penalty"):
        drm.fit_deregime(x, y, gate_penalty=-1.0)
    with pytest.raises(ValueError, match="weight_decay"):
        drm.fit_deregime(x, y, weight_decay=float("nan"))
    with pytest.raises(ValueError, match="2 \\* n_regimes"):
        drm.fit_deregime(x[:3], y[:3], n_regimes=2)


@requires_torch
def test_fit_deterministic_and_loss_decreases(
    small_fit: tuple[drm.SyntheticRegimeStream, drm.DeRegiMEResult],
) -> None:
    stream, res = small_fit
    again = drm.fit_deregime(stream.X, stream.Y, n_regimes=3, epochs=150, sigma_floor=0.1, seed=0)
    other = drm.fit_deregime(stream.X, stream.Y, n_regimes=3, epochs=150, sigma_floor=0.1, seed=1)
    assert np.array_equal(res.loss_curve, again.loss_curve)
    assert res.nlpd_train == again.nlpd_train
    assert not np.array_equal(res.loss_curve, other.loss_curve)
    assert res.loss_curve[-1] < res.loss_curve[0]  # training reduces the score
    # nlpd_train is computed via the numpy predict path: it must reproduce
    assert res.nlpd_train == pytest.approx(res.nlpd(stream.X, stream.Y), abs=1e-12)


@requires_torch
def test_prediction_shapes_simplex_and_learned_kernel_psd(
    small_fit: tuple[drm.SyntheticRegimeStream, drm.DeRegiMEResult],
) -> None:
    stream, res = small_fit
    pred = res.predict(stream.X)
    n, h = stream.Y.shape
    assert pred.mu.shape == (n, h) and pred.delta_mean.shape == (n, h)
    assert pred.delta_var.shape == (n, h) and np.all(pred.delta_var >= 0.0)
    assert pred.scales.shape == (n, h, 3) and np.all(pred.scales > 0.0)
    assert pred.gates.shape == (n, h, 3)
    assert np.allclose(pred.gates.sum(axis=-1), 1.0, atol=1e-6)
    assert pred.tails.shape == (3,) and np.all(pred.tails >= 2.0)  # nu_min floor
    assert 1 <= pred.effective_regimes() <= 3
    # propriety of the LEARNED density at three locations (quadrature)
    for i in (0, n // 2, n - 1):
        s_max = float(pred.scales[i].max())
        grid = np.linspace(-12.0 * (1.0 + s_max), 12.0 * (1.0 + s_max), 20_001)[:, None]
        dens = np.exp(
            drm.mixture_log_density(
                np.repeat(grid, h, axis=1),
                np.repeat(pred.mu[i][None, :], grid.shape[0], axis=0),
                delta_mean=np.repeat(pred.delta_mean[i][None, :], grid.shape[0], axis=0),
                delta_var=np.repeat(pred.delta_var[i][None, :], grid.shape[0], axis=0),
                scales=np.repeat(pred.scales[i][None, :, :], grid.shape[0], axis=0),
                tails=pred.tails,
                gates=np.repeat(pred.gates[i][None, :, :], grid.shape[0], axis=0),
            )
        )
        integral = float(np.trapezoid(dens[:, 0], grid[:, 0]))
        # trapezoid under-integrates the heavy Student-t tails beyond the grid;
        # the bound is the quadrature floor — propriety itself is exact by
        # Tonelli (mixture of proper components), so a real defect would miss
        # by orders of magnitude, not 1e-4.
        assert abs(integral - 1.0) < 5e-4
    # learned regime-mixing kernel is PSD (paper Theorem 2 on fitted params)
    k = res.mix_kernel(stream.X[:40], horizon=0)
    assert k.shape == (40, 40) and np.allclose(k, k.T, atol=1e-12)
    assert np.linalg.eigvalsh(k).min() >= -1e-9
    with pytest.raises(ValueError, match="horizon"):
        res.mix_kernel(stream.X[:5], horizon=h)
    with pytest.raises(ValueError, match="features"):
        res.predict(stream.X[:, :-1])


@requires_torch
def test_softmax_gate_and_gaussian_family_variants_train() -> None:
    stream = drm.simulate_regime_stream(
        500, sigmas=(0.5, 1.5), n_horizons=1, p_stay=0.98, tail_nu=None, seed=3
    )
    sm = drm.fit_deregime(
        stream.X, stream.Y, n_regimes=3, gate="softmax", epochs=80, sigma_floor=0.1, seed=0
    )
    pred = sm.predict(stream.X)
    assert np.allclose(pred.gates.sum(axis=-1), 1.0, atol=1e-6)
    assert 1 <= pred.effective_regimes() <= 3
    assert np.isfinite(sm.nlpd(stream.X, stream.Y))
    ga = drm.fit_deregime(
        stream.X, stream.Y, n_regimes=2, family="gaussian", epochs=80, sigma_floor=0.1, seed=0
    )
    gpred = ga.predict(stream.X)
    assert np.allclose(gpred.tails, 1.0)  # ignored placeholder for the Gaussian family
    assert np.isfinite(ga.nlpd(stream.X, stream.Y))


@requires_torch
def test_synthetic_benchmark_beats_ngboost_and_recovers_regime_structure() -> None:
    """Headline SYNTHETIC validation (lane adaptation of the paper's NLPD claim).

    Fixed protocol (seed=0 train stream, eval_seed=101 scoring stream, 3
    planted regimes with sigma=(0.3, 0.8, 1.6), R_max=4 truncation):
    DeRegiME NLPD <= NGBoostGaussian NLPD within the documented MC
    tolerance, effective regime count within one of the planted count, and
    per-regime scales recovered by nearest-neighbour matching (labels are
    identified only up to permutation, paper Appendix B.5). Observed at
    authoring time: gap about -0.33 nats, R_eff = 4, scale error about
    0.17. SYNTHETIC correctness evidence, never market evidence.
    """
    bench = drm.synthetic_regime_benchmark(
        sigmas=PLANTED_SIGMAS, n_regimes=4, seed=0, eval_seed=101
    )
    m = bench.metrics
    expected_keys = {
        "drm_nlpd",
        "drm_ngboost_nlpd",
        "drm_nlpd_gap_vs_ngboost",
        "drm_nlpd_train",
        "drm_effective_regimes",
        "drm_planted_regimes",
        "drm_regime_scale_max_rel_err",
        "drm_nu_recovered_min",
        "drm_seed",
        "drm_n_eval",
        "drm_n_horizons",
    }
    assert set(m) == expected_keys
    assert all(math.isfinite(v) for v in m.values())
    # (a) NLPD win over NGBoostGaussian within the documented tolerance
    assert m["drm_nlpd"] <= m["drm_ngboost_nlpd"] + NLPD_TOLERANCE, (
        f"DeRegiME NLPD {m['drm_nlpd']:.3f} did not beat NGBoost "
        f"{m['drm_ngboost_nlpd']:.3f} within tolerance {NLPD_TOLERANCE}"
    )
    assert m["drm_nlpd_gap_vs_ngboost"] == pytest.approx(
        m["drm_nlpd"] - m["drm_ngboost_nlpd"], abs=1e-12
    )
    # (b) recovered regime count close to planted (paper R_eff diagnostic)
    assert m["drm_planted_regimes"] == 3.0
    assert abs(m["drm_effective_regimes"] - 3.0) <= 1.0
    # (c) per-regime scale recovery
    assert m["drm_regime_scale_max_rel_err"] <= SCALE_RECOVERY_TOL
    # (d) learned tails stay above the finite-variance floor nu_min = 2
    assert m["drm_nu_recovered_min"] > 2.0
    # eval prediction is on the held-out stream, not the training batch
    assert m["drm_n_eval"] > 1000.0
