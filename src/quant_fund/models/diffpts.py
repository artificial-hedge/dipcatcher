"""DiffPTS: full-ELBO conditional denoising diffusion for probabilistic forecasting.

Ye, W., Li, D., Liu, H., Jiang, H., Sekimoto, Y. & Jiang, R. (2026),
"DiffPTS: Rethinking Diffusion ELBO for Probabilistic Time Series
Forecasting", NeurIPS 2026, arXiv:2609.32363. DiffPTS recasts conditional
DDPM forecasting under a Location-Scale Noise Model (LSNM): the noising
process is defined on the STANDARDIZED target ``z_0 = (y - mu_phi(x)) /
sigma_phi(x)`` where ``mu_phi`` / ``sigma_phi`` are learned location/scale
estimators, i.e. in y-space

    q(y_t | y_0, x) = N( sqrt(abar_t) y_0 + (1 - sqrt(abar_t)) mu_phi(x),
                         sigma_phi(x)^2 (1 - abar_t) ),

so the terminal prior is the learned ``N(mu_phi, sigma_phi^2)`` rather than
``N(0, I)`` — this is what accommodates distributional shift (the paper's
motivation over CSDI/TimeGrad-style paradigms, which it unifies). Rewriting
the DDPM evidence lower bound under the LSNM, DiffPTS shows the ELBO
naturally induces a Gaussian negative log likelihood objective for the
location-scale estimators, so mean/variance estimation and the denoiser are
trained END-TO-END inside one variational objective instead of as a separate
regression. The implemented joint loss keeps that mechanism verbatim:

    L = E_{t,eps} || eps - eps_theta(z_t, t, x) ||^2            (denoising)
        + lambda_nll * [ 0.5 log(2 pi) + log sigma_phi(x)
                         + 0.5 ((y - mu_phi(x)) / sigma_phi(x))^2 ],  (induced NLL)

with ``z_t = sqrt(abar_t) z_0 + sqrt(1 - abar_t) eps`` computed through
``mu_phi``/``sigma_phi`` so gradients flow jointly. The ``L_T`` KL term is
parameter-free under the fixed schedule (the prior depends on the estimators
but q(y_T|y_0) converges to the same family; see Deviations). References for
the machinery: Ho, J., Jain, A. & Abbeel, P. (2020), "Denoising Diffusion
Probabilistic Models", NeurIPS 33, arXiv:2006.11239 (forward process,
epsilon-parameterization, "simple" uniform weighting); Song, J., Meng, C. &
Ermon, S. (2021), "Denoising Diffusion Implicit Models", ICLR 2021,
arXiv:2010.02502 (DDIM deterministic few-step sampler, eta = 0); Nichol, A.
& Dhariwal, P. (2021), "Improved Denoising Diffusion Probabilistic Models",
ICML 2021, arXiv:2102.09672 (cosine variance schedule); Rasul, K. et al.
(2021), "Autoregressive Denoising Diffusion Models for Multivariate
Probabilistic Time Series Forecasting" (TimeGrad), ICML 2021,
arXiv:2101.12072, and Tashiro, Y. et al. (2021), "CSDI: Conditional
Score-based Diffusion Models for Probabilistic Time Series Imputation",
NeurIPS 34, arXiv:2107.03502 — the two paradigms the LSNM ELBO unifies.

Benchmark-honesty references followed here: "StocBench: A Benchmark for
Generative Modeling of Stochastic Dynamics", arXiv:2608.22309 (2026) —
scores are reported as a FUNCTION of sampler budget (number of function
evaluations, NFE); and Greenbury, S.F., Jersakova, R., Conti, P., Famili,
M., Sprague, C.I., Brown, E. & McEwen, J.D. (2026), "Reliability of
Probabilistic Emulation of Physical Systems", arXiv:2606.12997, whose
central finding is that CRPS-trained ensembles typically achieve more
reliable coverage than latent-space generative models at matched budget.
(The lane spec cited the last paper under the name "AutoCast"; the arXiv id
resolves to the Greenbury et al. paper above and its finding is the intended
one — the citation is corrected here and in the structured report.)
Following it, ``evaluate_synthetic_stream`` REPORTS the DiffPTS-vs-baseline
delta honestly rather than asserting diffusion wins.

Deviations from the paper (documented, harness-driven):

1. Univariate single-step target conditioned on a lookback feature matrix —
   the C = H = 1 case on the harness ``(X, y)`` contract (``make_windows``
   builds the windows from a raw series). The paper's multivariate
   multi-horizon conditioning reduces to this canonically.
2. Objective: uniform ("simple") epsilon-matching weights plus the induced
   Gaussian NLL, instead of the paper's per-term ELBO weights. DDPM §3.4
   shows the uniform weighting is itself a reweighted ELBO; keeping the
   induced NLL term preserves the paper's actual contribution — joint,
   end-to-end location-scale training. The ``L_T`` KL term is parameter-free
   under the fixed beta schedule and is dropped (standard DDPM practice).
3. Location-scale heads: ``mu_phi`` is a Linear head warm-started with the
   EXACT closed-form ridge forecast (``RidgeMeanForecaster`` — the paper's
   "pretrained mean estimator" role, reused from
   ``models/odd_residual_flows``, not reimplemented); ``log sigma_phi`` is a
   bounded Linear head ``log_sigma_base + B tanh(raw / B)`` initialized at
   the training residual std. Arbitrary pretrained forecasters (the paper's
   option) can replace the ridge by passing ``stage1``.
4. Denoiser: a small tanh MLP over [z_t, sinusoidal time features,
   standardized lookback context] — the paper's task-specific backbones are
   replaced by the harness-minimal MLP (the TORF sibling's ablation finds
   MLPs adequate at this scale, arXiv:2608.11114 Table 5).
5. Inference is pure numpy on extracted float64 parameters (mirrors
   ``models/odd_residual_flows``): torch is needed ONLY by ``fit``. This
   goes further than the spec requires and keeps the DDPM ancestral sampler,
   the DDIM few-step sampler, quantiles, PIT and CRPS all torch-free.
6. Sampling-time ``x0_clip`` static thresholding (default 5.0 in standardized
   space): the cosine schedule's ``abar_T ~ 1e-6 .. 1e-5`` makes the ``1/sqrt(abar)``
   eps-decodage explode under any imperfect denoiser, and the excursion
   self-sustains across the reverse trajectory. The cheap ancestor of
   Imagen's dynamic thresholding (Saharia et al. 2022, arXiv:2205.11487);
   ``x0_from_epsilon`` exposes the exact unclipped form for the algebraic
   tests, and ``clip`` is fail-closed validated.

Honesty (AGENTS.md contract): every number this module produces is a proper
score (CRPS via ``crps_empirical``, pinball, PIT histogram / KS) on seeded
SYNTHETIC streams — correctness evidence, never market evidence; the bench
dict carries ``dgp="fixture"`` / ``claim="research_metric_only"`` /
``synthetic=...`` markers. No Sharpe/Sortino/Calmar/P&L headline is
computed; there is no live-trading claim and no broker connectivity.

Composition (reuse, do not reimplement): ``metrics.scoring.crps_empirical``
and ``metrics.scoring.mean_pinball`` and ``metrics.probability.pit_ks`` own
the scores; ``models/ngboost_lite.NGBoostGaussian`` and ``models/qrf.
QuantileRegressionForest`` are the proper-score baselines in
``evaluate_synthetic_stream``; ``RidgeMeanForecaster`` is imported for the
LSNM warm start. The sibling flow head ``models/odd_residual_flows.TORF``
is DIFFERENT in kind: a deterministic change-of-variables density with an
analytic CDF and an exactly preserved conditional mean, versus this module's
stochastic posterior sampler — no closed-form density exists here, so CRPS
is the Monte Carlo ``crps_empirical`` over sampler draws, and results are
reported per sampler budget (StocBench-style) rather than as a single
number.

Conventions: torch is the optional ``nn`` extra, imported lazily via
:func:`_torch` (mirrors ``models/deep_hedging.py``), so the module imports
cleanly without torch and ``fit`` raises ``ImportError`` with guidance.
Training is CPU single-thread full-batch Adam; every randomness source
(torch init, per-epoch (t, eps) draws, sampler noise) is seeded, so repeated
calls are bit-identical (GPU determinism not claimed). Fail-closed: invalid
hyperparameters, non-finite or mismatched inputs, too-few samples,
zero-variance targets, unfitted use, out-of-range timesteps, and non-finite
training loss all raise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.probability import pit_ks
from quant_fund.metrics.scoring import crps_empirical, mean_pinball
from quant_fund.models.odd_residual_flows import RidgeMeanForecaster

Array = NDArray[np.float64]

__all__ = [
    "DiffPTSModel",
    "DiffPTSFitInfo",
    "DiffusionParams",
    "DiffusionSamples",
    "DiffusionSchedule",
    "SampleScores",
    "ddim_sample",
    "ddpm_ancestral_sample",
    "ddim_step",
    "evaluate_samples",
    "evaluate_synthetic_stream",
    "make_windows",
    "posterior_mean_variance",
    "predict_epsilon",
    "predict_location_scale",
    "q_sample",
    "simulate_synthetic_stream",
    "time_features",
    "x0_from_epsilon",
]

_Layers = tuple[tuple[Array, Array], ...]

_LOG_2PI = math.log(2.0 * math.pi)
_MIN_FIT_SAMPLES = 16


def _torch() -> Any:
    """Lazily import torch (optional ``nn`` extra); fail closed with guidance."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "diffpts needs the optional 'nn' extra (torch): uv sync --extra nn"
        ) from exc
    return torch


def _check_count(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be an int >= 1; got {value!r}")
    return int(value)


def _check_positive(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite; got {value!r}")
    return v


def _check_matrix(X: Array, name: str) -> Array:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n, p); got ndim={arr.ndim}")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} must be non-empty")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    return arr


def _check_vector(v: Array, n: int, name: str) -> Array:
    arr = np.asarray(v, dtype=float).ravel()
    if arr.shape[0] != n:
        raise ValueError(f"{name} must have length {n}; got {arr.shape[0]}")
    if arr.size and not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    return arr


# ---------------------------------------------------------------------------
# Forward process: variance schedule and closed-form conditionals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffusionSchedule:
    """Variance schedule of the forward noising process.

    ``betas`` = (beta_1, ..., beta_T) with each beta_t in (0, 1);
    ``alphas_cumprod`` = abar_t = prod_{s<=t} (1 - beta_s), strictly
    decreasing into (0, 1). ``abar(0) := 1`` (DDPM convention).
    """

    betas: Array
    alphas_cumprod: Array

    def __post_init__(self) -> None:
        b = np.asarray(self.betas, dtype=float).reshape(-1)
        a = np.asarray(self.alphas_cumprod, dtype=float).reshape(-1)
        if b.size < 1:
            raise ValueError("betas must be non-empty (n_steps >= 1)")
        if not bool(np.all(np.isfinite(b))) or bool(np.any((b <= 0.0) | (b >= 1.0))):
            raise ValueError("betas must be finite and inside (0, 1)")
        expected = np.cumprod(1.0 - b)
        if a.shape != b.shape or not bool(np.allclose(a, expected, rtol=1e-9, atol=1e-12)):
            raise ValueError("alphas_cumprod must equal cumprod(1 - betas)")
        if not bool(np.all(np.diff(a) < 0.0)) or not (0.0 < a[-1] < 1.0):
            raise ValueError("alphas_cumprod must be strictly decreasing into (0, 1)")
        object.__setattr__(self, "betas", b)
        object.__setattr__(self, "alphas_cumprod", a)

    @classmethod
    def linear(
        cls, n_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2
    ) -> DiffusionSchedule:
        """Linear beta grid of Ho et al. (2020)."""
        t = _check_count(n_steps, "n_steps")
        lo = _check_positive(beta_start, "beta_start")
        hi = _check_positive(beta_end, "beta_end")
        if not lo < hi < 1.0:
            raise ValueError("need 0 < beta_start < beta_end < 1")
        betas = np.linspace(lo, hi, t, dtype=float)
        return cls(betas=betas, alphas_cumprod=np.cumprod(1.0 - betas))

    @classmethod
    def cosine(cls, n_steps: int, s: float = 0.008) -> DiffusionSchedule:
        """Cosine abar schedule of Nichol & Dhariwal (2021), betas clipped < 0.999.

        ``abar_t = f(t) / f(0)`` with ``f(t) = cos^2((t/T + s)/(1 + s) * pi/2)``;
        ``beta_t = 1 - abar_t / abar_{t-1}`` clipped below 0.999, then abar is
        rebuilt from the clipped betas so the pair stays exactly consistent.
        """
        t = _check_count(n_steps, "n_steps")
        off = float(s)
        if not math.isfinite(off) or off < 0.0 or off >= 1.0:
            raise ValueError(f"cosine offset s must be in [0, 1); got {s!r}")

        def f(k: Array) -> Array:
            return np.asarray(np.cos((k / t + off) / (1.0 + off) * math.pi / 2.0) ** 2)

        kgrid = np.arange(t + 1, dtype=float)
        abar = f(kgrid) / f(np.zeros(1))[0]
        betas = np.clip(1.0 - abar[1:] / abar[:-1], 1e-5, 0.999)
        return cls(betas=betas, alphas_cumprod=np.cumprod(1.0 - betas))

    @property
    def n_steps(self) -> int:
        return int(self.betas.shape[0])

    def alpha_bar(self, t: int | Array) -> float | Array:
        """abar_t for t in {0, ..., n_steps}; abar_0 := 1 (DDPM convention)."""
        arr = np.asarray(t, dtype=float)
        if (
            not bool(np.all(np.isfinite(arr)))
            or bool(np.any(arr < 0))
            or bool(np.any(arr > self.n_steps))
        ):
            raise ValueError(f"t must be in [0, {self.n_steps}]; got {t!r}")
        ti = np.asarray(arr, dtype=np.int64)
        vals = np.where(ti >= 1, self.alphas_cumprod[np.maximum(ti, 1) - 1], 1.0)
        if np.ndim(t) == 0:
            return float(vals)
        return np.asarray(vals, dtype=float)


def q_sample(z0: Array, t: int | Array, eps: Array, schedule: DiffusionSchedule) -> Array:
    """Closed-form forward marginal ``z_t = sqrt(abar_t) z_0 + sqrt(1 - abar_t) eps``.

    ``t`` is a scalar or a per-row array of timesteps (training samples one t
    per observation). Fail-closed on shape mismatch or t outside [0, T].
    """
    z = np.asarray(z0, dtype=float).reshape(-1)
    e = np.asarray(eps, dtype=float).reshape(-1)
    if z.shape != e.shape:
        raise ValueError(f"z0 and eps must have the same shape; got {z.shape} vs {e.shape}")
    if z.size and (not bool(np.all(np.isfinite(z))) or not bool(np.all(np.isfinite(e)))):
        raise ValueError("z0 and eps must be finite")
    ab = np.asarray(schedule.alpha_bar(t), dtype=float).reshape(-1)
    if ab.size == 1:
        ab = np.full(z.shape, float(ab[0]))
    if ab.shape != z.shape:
        raise ValueError(f"t must broadcast to z0's shape {z.shape}; got {ab.shape}")
    return np.asarray(np.sqrt(ab) * z + np.sqrt(1.0 - ab) * e, dtype=float)


def x0_from_epsilon(
    z_t: Array,
    eps_hat: Array,
    t: int | Array,
    schedule: DiffusionSchedule,
    *,
    clip: float | None = None,
) -> Array:
    """DDPM posterior decodage: ``x0_hat = (z_t - sqrt(1 - abar_t) eps_hat) / sqrt(abar_t)``.

    ``clip`` is an optional static threshold applied to ``x0_hat``: as
    ``abar_t -> 0`` the ``1 / sqrt(abar_t)`` gain turns tiny eps errors into
    arbitrary ``x0_hat`` excursions, which then self-sustain through the
    reverse trajectory. Static thresholding (the cheap ancestor of Imagen's
    dynamic thresholding, Saharia et al. 2022, arXiv:2205.11487) clamps the
    estimate to a plausible standardized-data range. ``clip=None`` keeps the
    exact closed form (used by the algebraic tests).
    """
    z = np.asarray(z_t, dtype=float).reshape(-1)
    e = np.asarray(eps_hat, dtype=float).reshape(-1)
    if z.shape != e.shape:
        raise ValueError(f"z_t and eps_hat must have the same shape; got {z.shape} vs {e.shape}")
    if not bool(np.all(np.isfinite(z))) or not bool(np.all(np.isfinite(e))):
        raise ValueError("z_t and eps_hat must be finite")
    ab = np.asarray(schedule.alpha_bar(t), dtype=float).reshape(-1)
    if ab.size == 1:
        ab = np.full(z.shape, float(ab[0]))
    if ab.shape != z.shape:
        raise ValueError(f"t must broadcast to z_t's shape {z.shape}; got {ab.shape}")
    x0 = (z - np.sqrt(1.0 - ab) * e) / np.sqrt(ab)
    if clip is not None:
        c = float(clip)
        if not math.isfinite(c) or c <= 0.0:
            raise ValueError(f"clip must be positive and finite; got {clip!r}")
        x0 = np.clip(x0, -c, c)
    return np.asarray(x0, dtype=float)


def posterior_mean_variance(
    z_t: Array, x0_hat: Array, t: int, schedule: DiffusionSchedule
) -> tuple[Array, float]:
    """Closed-form ``q(z_{t-1} | z_t, z_0)`` of Ho et al. (2020, eqs. 6-7).

    Scalar ``t`` in {1, ..., T}: ``mu~ = (sqrt(abar_{t-1}) beta_t /
    (1 - abar_t)) x0 + (sqrt(alpha_t) (1 - abar_{t-1}) / (1 - abar_t)) z_t``
    and ``beta_t~ = beta_t (1 - abar_{t-1}) / (1 - abar_t)``. At ``t = 1``
    ``abar_0 = 1`` gives ``beta_1~ = 0`` and ``mu~ = x0`` exactly — the
    terminal step is deterministic.
    """
    ti = _check_count(t, "t")
    if ti > schedule.n_steps:
        raise ValueError(f"t must be <= n_steps={schedule.n_steps}; got {t}")
    z = np.asarray(z_t, dtype=float).reshape(-1)
    x0 = np.asarray(x0_hat, dtype=float).reshape(-1)
    if z.shape != x0.shape:
        raise ValueError(f"z_t and x0_hat must have the same shape; got {z.shape} vs {x0.shape}")
    if not bool(np.all(np.isfinite(z))) or not bool(np.all(np.isfinite(x0))):
        raise ValueError("z_t and x0_hat must be finite")
    ab_t = float(schedule.alpha_bar(ti))
    ab_prev = float(schedule.alpha_bar(ti - 1)) if ti > 1 else 1.0
    beta_t = float(schedule.betas[ti - 1])
    coef_x0 = math.sqrt(ab_prev) * beta_t / (1.0 - ab_t)
    coef_zt = math.sqrt(1.0 - beta_t) * (1.0 - ab_prev) / (1.0 - ab_t)
    mean = np.asarray(coef_x0 * x0 + coef_zt * z, dtype=float)
    var = beta_t * (1.0 - ab_prev) / (1.0 - ab_t)
    return mean, float(var)


def ddim_step(
    z_t: Array,
    eps_hat: Array,
    t: int,
    t_prev: int,
    schedule: DiffusionSchedule,
    *,
    eta: float = 0.0,
    noise: Array | None = None,
    clip: float | None = None,
) -> Array:
    """One DDIM reverse step ``z_t -> z_{t_prev}`` (Song et al. 2021, eq. 12).

    ``eta = 0`` is the deterministic sampler; ``t_prev = 0`` lands on
    ``abar_0 = 1``, returning ``x0_hat`` (sigma = 0 there). ``noise`` supplies
    the stochastic draw when ``eta > 0`` and is ignored at ``eta = 0``.
    """
    ti = _check_count(t, "t")
    if ti > schedule.n_steps:
        raise ValueError(f"t must be <= n_steps={schedule.n_steps}; got {t}")
    tp = int(t_prev)
    if tp < 0 or tp >= ti:
        raise ValueError(f"t_prev must be in [0, t); got {t_prev!r}")
    e = float(eta)
    if not math.isfinite(e) or not (0.0 <= e <= 1.0):
        raise ValueError(f"eta must be in [0, 1]; got {eta!r}")
    z = np.asarray(z_t, dtype=float).reshape(-1)
    eps = np.asarray(eps_hat, dtype=float).reshape(-1)
    if z.shape != eps.shape:
        raise ValueError(f"z_t and eps_hat must have the same shape; got {z.shape} vs {eps.shape}")
    ab_t = float(schedule.alpha_bar(ti))
    ab_prev = float(schedule.alpha_bar(tp)) if tp > 0 else 1.0
    x0 = x0_from_epsilon(z, eps, ti, schedule, clip=clip)
    sigma = e * math.sqrt((1.0 - ab_prev) / (1.0 - ab_t)) * math.sqrt(1.0 - ab_t / ab_prev)
    c = math.sqrt(max(1.0 - ab_prev - sigma * sigma, 0.0))
    nz = np.zeros_like(z)
    if sigma > 0.0:
        if noise is None:
            raise ValueError("noise is required when eta > 0")
        nz = np.asarray(noise, dtype=float).reshape(-1)
        if nz.shape != z.shape or not bool(np.all(np.isfinite(nz))):
            raise ValueError("noise must match z_t's shape and be finite")
    return np.asarray(math.sqrt(ab_prev) * x0 + c * eps + sigma * nz, dtype=float)


# ---------------------------------------------------------------------------
# Numpy inference core: extracted float64 parameters (torch-free)
# ---------------------------------------------------------------------------


def _np_mlp(F: Array, layers: _Layers) -> Array:
    """Numpy mirror of the torch MLPs: tanh hidden layers, linear output."""
    z = F
    last = len(layers) - 1
    for i, (W, b) in enumerate(layers):
        z = z @ W.T + b
        if i < last:
            z = np.tanh(z)
    return np.asarray(z, dtype=float)


def time_features(t: Array | int, n_steps: int, n_freqs: int) -> Array:
    """Deterministic timestep embedding ``(k, 1 + 2 * n_freqs)``.

    Features: the normalized time ``t / T`` plus sin/cos at dyadic angular
    frequencies ``2 pi * 2^k * t / T`` for ``k = 0..n_freqs-1`` — a fixed
    transform (no learned table), identical in the torch and numpy paths.
    """
    tt = np.asarray(t, dtype=float).reshape(-1)
    n = _check_count(n_steps, "n_steps")
    nf = _check_count(n_freqs, "n_freqs")
    if not bool(np.all(np.isfinite(tt))) or bool(np.any(tt < 0)) or bool(np.any(tt > n)):
        raise ValueError(f"t must be finite and inside [0, {n}]; got {t!r}")
    tt = tt / float(n)
    cols: list[Array] = [tt]
    for k in range(nf):
        ang = 2.0 * math.pi * (2.0**k) * tt
        cols.append(np.sin(ang))
        cols.append(np.cos(ang))
    return np.asarray(np.stack(cols, axis=1), dtype=float)


def _check_layers(layers: _Layers, d_in: int, d_out: int, name: str) -> _Layers:
    out: list[tuple[Array, Array]] = []
    prev = int(d_in)
    for i, (W, b) in enumerate(layers):
        wa = np.asarray(W, dtype=float)
        ba = np.asarray(b, dtype=float).reshape(-1)
        if wa.ndim != 2 or wa.shape[1] != prev:
            raise ValueError(f"{name} layer {i} weight must be (out, {prev}); got {wa.shape}")
        if ba.shape != (wa.shape[0],):
            raise ValueError(f"{name} layer {i} bias must be ({wa.shape[0]},); got {ba.shape}")
        if not bool(np.all(np.isfinite(wa))) or not bool(np.all(np.isfinite(ba))):
            raise ValueError(f"{name} layer {i} must be finite")
        out.append((wa, ba))
        prev = int(wa.shape[0])
    if not out or prev != d_out:
        raise ValueError(f"{name} must be non-empty with output dim {d_out}")
    return tuple(out)


@dataclass(frozen=True, eq=False)
class DiffusionParams:
    """Float64 inference parameters extracted from a torch ``fit``.

    ``loc_layers`` / ``scale_layers`` are the LSNM heads on standardized
    context ``F``: ``mu = loc(F)`` and ``log sigma = log_sigma_base +
    log_sigma_bound * tanh(scale(F) / log_sigma_bound)``. ``denoiser_layers``
    maps ``[z_t, time_features(t), F] -> eps_hat``. All inference is pure
    numpy — an instance may be constructed by hand for sampler tests.
    """

    denoiser_layers: _Layers
    loc_layers: _Layers
    scale_layers: _Layers
    ctx_mean: Array
    ctx_std: Array
    log_sigma_base: float
    log_sigma_bound: float
    schedule: DiffusionSchedule
    n_time_freqs: int
    x0_clip: float = 5.0

    def __post_init__(self) -> None:
        cm = np.asarray(self.ctx_mean, dtype=float).reshape(-1)
        cs = np.asarray(self.ctx_std, dtype=float).reshape(-1)
        if cm.size < 1 or cm.shape != cs.shape:
            raise ValueError("ctx_mean/ctx_std must be matching non-empty vectors")
        if not bool(np.all(np.isfinite(cm))) or not bool(np.all(np.isfinite(cs))):
            raise ValueError("ctx_mean/ctx_std must be finite")
        if bool(np.any(cs <= 0.0)):
            raise ValueError("ctx_std must be strictly positive")
        nf = _check_count(self.n_time_freqs, "n_time_freqs")
        p = int(cm.size)
        d_in = 1 + (1 + 2 * nf) + p
        object.__setattr__(
            self, "denoiser_layers", _check_layers(self.denoiser_layers, d_in, 1, "denoiser_layers")
        )
        object.__setattr__(self, "loc_layers", _check_layers(self.loc_layers, p, 1, "loc_layers"))
        object.__setattr__(
            self, "scale_layers", _check_layers(self.scale_layers, p, 1, "scale_layers")
        )
        if not isinstance(self.schedule, DiffusionSchedule):
            raise ValueError("schedule must be a DiffusionSchedule")
        xc = float(self.x0_clip)
        if not math.isfinite(xc) or xc <= 0.0:
            raise ValueError(f"x0_clip must be positive and finite; got {self.x0_clip!r}")
        object.__setattr__(self, "x0_clip", xc)
        lb = float(self.log_sigma_base)
        if not math.isfinite(lb):
            raise ValueError("log_sigma_base must be finite")
        object.__setattr__(self, "log_sigma_base", lb)
        object.__setattr__(
            self, "log_sigma_bound", _check_positive(self.log_sigma_bound, "log_sigma_bound")
        )
        object.__setattr__(self, "ctx_mean", cm)
        object.__setattr__(self, "ctx_std", cs)
        object.__setattr__(self, "n_time_freqs", nf)


def _context(params: DiffusionParams, X: Array) -> Array:
    arr = _check_matrix(X, "X")
    if arr.shape[1] != params.ctx_mean.size:
        raise ValueError(
            f"X feature count mismatch: params hold p={params.ctx_mean.size}, got {arr.shape[1]}"
        )
    return np.asarray((arr - params.ctx_mean) / params.ctx_std, dtype=float)


def predict_location_scale(params: DiffusionParams, X: Array) -> tuple[Array, Array]:
    """LSNM heads: ``(mu_phi(x), sigma_phi(x))`` for each row of ``X`` (numpy)."""
    F = _context(params, X)
    mu = _np_mlp(F, params.loc_layers)[:, 0]
    raw = _np_mlp(F, params.scale_layers)[:, 0]
    b = params.log_sigma_bound
    log_sigma = params.log_sigma_base + b * np.tanh(raw / b)
    sigma = np.asarray(np.exp(log_sigma), dtype=float)
    return np.asarray(mu, dtype=float), sigma


def _eps_hat(params: DiffusionParams, z_t: Array, t: int, F: Array) -> Array:
    """eps_theta on ALREADY-standardized context ``F`` at scalar timestep ``t``."""
    z = np.asarray(z_t, dtype=float).reshape(-1)
    tf = time_features(t, params.schedule.n_steps, params.n_time_freqs)
    tf_rep = np.repeat(tf, z.shape[0], axis=0)
    feats = np.concatenate([z[:, None], tf_rep, F], axis=1)
    return np.asarray(_np_mlp(feats, params.denoiser_layers)[:, 0], dtype=float)


def predict_epsilon(params: DiffusionParams, z_t: Array, t: int | Array, X: Array) -> Array:
    """Predicted noise ``eps_theta(z_t, t, x)`` for rows ``(z_t[i], t[i], X[i])``.

    ``t`` may be a scalar (shared timestep) or a per-row array. All numpy.
    """
    F = _context(params, X)
    z = np.asarray(z_t, dtype=float).reshape(-1)
    if z.shape[0] != F.shape[0]:
        raise ValueError(f"z_t length must match X rows {F.shape[0]}; got {z.shape[0]}")
    if not bool(np.all(np.isfinite(z))):
        raise ValueError("z_t must be finite")
    tt = np.asarray(t)
    if tt.ndim == 0:
        return _eps_hat(params, z, int(tt), F)
    tf = time_features(np.asarray(tt, dtype=float), params.schedule.n_steps, params.n_time_freqs)
    feats = np.concatenate([z[:, None], tf, F], axis=1)
    return np.asarray(_np_mlp(feats, params.denoiser_layers)[:, 0], dtype=float)


@dataclass(frozen=True)
class DiffusionSamples:
    """Sampler output: ``samples`` (n, M) in y-space, ``nfe`` denoiser evals.

    ``nfe`` (number of function evaluations per draw) is the sampler budget
    StocBench (arXiv:2608.22309) reports scores against: the DDPM ancestral
    sampler costs T evals, DDIM costs ``steps``.
    """

    samples: Array
    nfe: int
    sampler: str
    seed: int


def ddpm_ancestral_sample(
    params: DiffusionParams, X: Array, n_samples: int, *, seed: int = 0
) -> DiffusionSamples:
    """DDPM ancestral sampler (Ho et al. 2020): T reverse steps, NFE = T.

    Draws ``z_T ~ N(0, I)`` in standardized space, applies the closed-form
    posterior ``q(z_{t-1} | z_t, x0_hat)`` with Gaussian noise for t > 1
    (the t = 1 step is deterministic), and maps back through the LSNM:
    ``y = mu_phi(x) + sigma_phi(x) z_0``. Pure numpy.
    """
    m = _check_count(n_samples, "n_samples")
    sched = params.schedule
    F = _context(params, X)
    n = F.shape[0]
    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal(n * m)
    F_rep = np.repeat(F, m, axis=0)
    for t in range(sched.n_steps, 0, -1):
        eps_hat = _eps_hat(params, z, t, F_rep)
        x0 = x0_from_epsilon(z, eps_hat, t, sched, clip=params.x0_clip)
        mean, var = posterior_mean_variance(z, x0, t, sched)
        z = mean + (math.sqrt(var) * rng.standard_normal(z.shape[0]) if t > 1 else 0.0)
    mu, sigma = predict_location_scale(params, X)
    out = mu[:, None] + sigma[:, None] * z.reshape(n, m)
    return DiffusionSamples(
        samples=np.asarray(out, dtype=float),
        nfe=sched.n_steps,
        sampler="ddpm",
        seed=int(seed),
    )


def ddim_sample(
    params: DiffusionParams,
    X: Array,
    n_samples: int,
    *,
    steps: int,
    seed: int = 0,
    eta: float = 0.0,
) -> DiffusionSamples:
    """DDIM few-step sampler (Song et al. 2021): ``steps`` evals, NFE = steps.

    Timesteps are the rounded ``linspace(1, T, steps)`` subsequence visited
    descending, with a final landing at ``t_prev = 0`` (abar_0 = 1). With
    ``eta = 0`` (default) the trajectory is deterministic given the z_T draw —
    the only randomness is the seeded ``z_T ~ N(0, I)``. ``steps > T`` is
    clamped honestly to the full schedule (NFE then equals T).
    """
    m = _check_count(n_samples, "n_samples")
    k = _check_count(steps, "steps")
    e = float(eta)
    if not math.isfinite(e) or not (0.0 <= e <= 1.0):
        raise ValueError(f"eta must be in [0, 1]; got {eta!r}")
    sched = params.schedule
    F = _context(params, X)
    n = F.shape[0]
    ts = np.unique(np.rint(np.linspace(1.0, sched.n_steps, k + 1)[1:]).astype(np.int64))
    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal(n * m)
    F_rep = np.repeat(F, m, axis=0)
    nfe = 0
    for i in range(ts.size - 1, -1, -1):
        t = int(ts[i])
        t_prev = int(ts[i - 1]) if i > 0 else 0
        eps_hat = _eps_hat(params, z, t, F_rep)
        noise = rng.standard_normal(z.shape[0]) if e > 0.0 and t_prev > 0 else None
        z = ddim_step(z, eps_hat, t, t_prev, sched, eta=e, noise=noise, clip=params.x0_clip)
        nfe += 1
    mu, sigma = predict_location_scale(params, X)
    out = mu[:, None] + sigma[:, None] * z.reshape(n, m)
    return DiffusionSamples(
        samples=np.asarray(out, dtype=float), nfe=nfe, sampler="ddim", seed=int(seed)
    )


# ---------------------------------------------------------------------------
# Torch training core (lazy ``nn`` extra)
# ---------------------------------------------------------------------------


def _build_mlp(torch: Any, d_in: int, hidden: Sequence[int], d_out: int) -> Any:
    layers: list[Any] = []
    d = int(d_in)
    for hh in hidden:
        layers += [torch.nn.Linear(d, int(hh)), torch.nn.Tanh()]
        d = int(hh)
    layers.append(torch.nn.Linear(d, int(d_out)))
    return torch.nn.Sequential(*layers)


def _torch_time_features(torch: Any, t: Any, n_steps: int, n_freqs: int) -> Any:
    """Torch mirror of :func:`time_features` (identical math on tensors)."""
    tt = t.to(torch.float32) / float(n_steps)
    cols = [tt]
    for k in range(n_freqs):
        ang = 2.0 * math.pi * (2.0**k) * tt
        cols.append(torch.sin(ang))
        cols.append(torch.cos(ang))
    return torch.stack(cols, dim=1)


def _extract_layers(torch: Any, net: Any) -> _Layers:
    mods = [net] if isinstance(net, torch.nn.Linear) else list(net)
    out: list[tuple[Array, Array]] = []
    for module in mods:
        if isinstance(module, torch.nn.Linear):
            w = np.array(module.weight.detach().numpy(), dtype=float)
            b = np.array(module.bias.detach().numpy(), dtype=float)
            out.append((w, b))
    return tuple(out)


def _train(
    torch: Any,
    F: Array,
    y: Array,
    *,
    n_steps: int,
    abars: Array,
    hidden: Sequence[int],
    n_time_freqs: int,
    log_sigma_bound: float,
    est_nll_weight: float,
    epochs: int,
    lr: float,
    seed: int,
    ridge_w: Array,
    ridge_b: float,
    resid_std: float,
) -> tuple[Any, Any, Any, list[float], list[float], list[float], float]:
    """Full-batch Adam on the joint DiffPTS objective; deterministic given seed.

    The location-scale heads are warm-started to the exact ridge forecast
    ``(ridge_w, ridge_b)`` (the paper's pretrained-mean role) and the
    constant residual scale ``log resid_std``. Per-epoch
    ``t ~ Uniform{1..T}`` and ``eps ~ N(0, I)`` come from a seeded numpy
    Generator — the same draws regardless of torch version.
    """
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    p = int(F.shape[1])
    n = int(F.shape[0])
    d_in = 1 + (1 + 2 * int(n_time_freqs)) + p
    den = _build_mlp(torch, d_in, hidden, 1)
    loc = torch.nn.Linear(p, 1)
    scale = torch.nn.Linear(p, 1)
    with torch.no_grad():
        loc.weight.copy_(torch.as_tensor(ridge_w[None, :], dtype=torch.float32))
        loc.bias.fill_(float(ridge_b))
        scale.weight.zero_()
        scale.bias.zero_()
    log_sigma_base = math.log(resid_std)
    opt = torch.optim.Adam(
        list(den.parameters()) + list(loc.parameters()) + list(scale.parameters()),
        lr=float(lr),
    )
    F_t = torch.as_tensor(F, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    abar_t = torch.as_tensor(abars, dtype=torch.float32)
    rng = np.random.default_rng(int(seed) + 1)
    curve: list[float] = []
    denoise_curve: list[float] = []
    nll_curve: list[float] = []
    for _ in range(int(epochs)):
        tt_np = rng.integers(1, n_steps + 1, size=n)
        eps_np = rng.standard_normal(n)
        tt = torch.as_tensor(tt_np, dtype=torch.int64)
        eps = torch.as_tensor(eps_np, dtype=torch.float32)
        ab = abar_t[tt - 1]
        opt.zero_grad(set_to_none=True)
        mu = loc(F_t)[:, 0]
        raw = scale(F_t)[:, 0]
        log_sig = float(log_sigma_base) + float(log_sigma_bound) * torch.tanh(
            raw / float(log_sigma_bound)
        )
        sig = torch.exp(log_sig)
        z0 = (y_t - mu) / sig
        z_t = torch.sqrt(ab) * z0 + torch.sqrt(1.0 - ab) * eps
        tf = _torch_time_features(torch, tt.to(torch.float32), n_steps, n_time_freqs)
        feats = torch.cat([z_t[:, None], tf, F_t], dim=1)
        eps_hat = den(feats)[:, 0]
        denoise = torch.mean((eps - eps_hat) ** 2)
        nll = torch.mean(0.5 * _LOG_2PI + log_sig + 0.5 * z0 * z0)
        loss = denoise + float(est_nll_weight) * nll
        val = float(loss.detach().numpy())
        if not math.isfinite(val):
            raise ValueError("diffusion ELBO objective is not finite during training")
        curve.append(val)
        denoise_curve.append(float(denoise.detach().numpy()))
        nll_curve.append(float(nll.detach().numpy()))
        loss.backward()
        opt.step()
    return den, loc, scale, curve, denoise_curve, nll_curve, log_sigma_base


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffPTSFitInfo:
    """Training trace of :meth:`DiffPTSModel.fit`.

    ``loss_curve`` records the joint objective BEFORE each Adam update
    (entry 0 is at the ridge-warm-started init). ``denoise_curve`` is the
    uniform-weighted epsilon-matching term and ``est_nll_curve`` the induced
    Gaussian NLL of the location-scale estimators — the two ELBO components
    DiffPTS trains jointly (arXiv:2609.32363).
    """

    epochs: int
    loss_curve: list[float]
    denoise_curve: list[float]
    est_nll_curve: list[float]
    final_loss: float
    seed: int
    n_steps: int


class DiffPTSModel:
    """DiffPTS conditional denoising-diffusion forecaster (arXiv:2609.32363).

    ``fit`` (the only torch entry point) trains the LSNM location-scale heads
    and the denoiser jointly on the induced objective; every inference method
    (``predict_params``, ``sample``, ``predict_quantiles``, ``pit``,
    ``crps``) is pure numpy on the extracted parameters and runs without
    torch. All samplers are seeded; repeated calls are bit-identical.
    """

    def __init__(
        self,
        n_steps: int = 50,
        hidden: Sequence[int] = (32, 32),
        n_time_freqs: int = 4,
        log_sigma_bound: float = 4.0,
        est_nll_weight: float = 1.0,
        epochs: int = 200,
        lr: float = 8e-3,
        seed: int = 0,
        ridge_alpha: float = 1e-2,
        schedule_kind: str = "cosine",
        x0_clip: float = 5.0,
    ) -> None:
        self.n_steps = _check_count(n_steps, "n_steps")
        xc = float(x0_clip)
        if not math.isfinite(xc) or xc <= 0.0:
            raise ValueError(f"x0_clip must be positive and finite; got {x0_clip!r}")
        self.x0_clip = xc
        hidden_widths = tuple(int(hh) for hh in hidden)
        if not hidden_widths or any(hh < 1 for hh in hidden_widths):
            raise ValueError(
                f"hidden must be a non-empty sequence of positive widths; got {hidden!r}"
            )
        self.hidden = hidden_widths
        self.n_time_freqs = _check_count(n_time_freqs, "n_time_freqs")
        self.log_sigma_bound = _check_positive(log_sigma_bound, "log_sigma_bound")
        w = float(est_nll_weight)
        if not math.isfinite(w) or w < 0.0:
            raise ValueError(f"est_nll_weight must be finite and >= 0; got {est_nll_weight!r}")
        self.est_nll_weight = w
        self.epochs = _check_count(epochs, "epochs")
        self.lr = _check_positive(lr, "lr")
        self.seed = int(seed)
        self.ridge_alpha = float(ridge_alpha)
        if not math.isfinite(self.ridge_alpha) or self.ridge_alpha < 0.0:
            raise ValueError(f"ridge_alpha must be finite and >= 0; got {ridge_alpha!r}")
        if schedule_kind not in ("cosine", "linear"):
            raise ValueError(f"schedule_kind must be 'cosine' or 'linear'; got {schedule_kind!r}")
        self.schedule_kind = schedule_kind
        self._params: DiffusionParams | None = None
        self.fit_info: DiffPTSFitInfo | None = None

    @property
    def is_fitted(self) -> bool:
        return self._params is not None

    @property
    def params(self) -> DiffusionParams:
        """The extracted numpy parameters (torch-free inference state)."""
        if self._params is None:
            raise RuntimeError("DiffPTSModel is not fitted")
        return self._params

    def _schedule(self) -> DiffusionSchedule:
        if self.schedule_kind == "cosine":
            return DiffusionSchedule.cosine(self.n_steps)
        return DiffusionSchedule.linear(self.n_steps)

    def fit(self, X: Array, y: Array) -> DiffPTSModel:
        """Jointly train the LSNM heads and denoiser on the ELBO objective.

        Inputs are validated BEFORE torch is touched, so input-contract
        errors raise ``ValueError`` even without the ``nn`` extra; a
        successful fit needs torch and raises ``ImportError`` with guidance
        when absent.
        """
        Xm = _check_matrix(X, "X")
        yv = _check_vector(y, Xm.shape[0], "y")
        if Xm.shape[0] < _MIN_FIT_SAMPLES:
            raise ValueError(f"too few samples: need >= {_MIN_FIT_SAMPLES}, got {Xm.shape[0]}")
        sd = float(np.std(yv))
        if not math.isfinite(sd) or sd <= 0.0:
            raise ValueError("y has zero variance; nothing to fit")
        ctx_mean = Xm.mean(axis=0)
        ctx_std = Xm.std(axis=0)
        ctx_std = np.where(ctx_std > 0.0, ctx_std, 1.0)  # constant columns: unit scale
        F = np.asarray((Xm - ctx_mean) / ctx_std, dtype=float)
        sched = self._schedule()
        ridge = RidgeMeanForecaster(alpha=self.ridge_alpha).fit(F, yv)
        resid = yv - ridge.predict(F)
        resid_std = float(np.std(resid))
        if not math.isfinite(resid_std) or resid_std <= 0.0:
            raise ValueError("residuals around the ridge forecast have zero variance")
        # Recover the ridge's affine form F @ w + b via the public predict API
        # (predict is affine in its input; evaluating it on the identity basis
        # and the zero vector yields the coefficients without private access).
        p = int(F.shape[1])
        ridge_b = float(ridge.predict(np.zeros((1, p)))[0])
        ridge_w = np.asarray(ridge.predict(np.eye(p)) - ridge_b, dtype=float)
        torch = _torch()
        den, loc, scale, curve, dcurve, ncurve, log_sigma_base = _train(
            torch,
            F,
            yv,
            n_steps=self.n_steps,
            abars=sched.alphas_cumprod,
            hidden=self.hidden,
            n_time_freqs=self.n_time_freqs,
            log_sigma_bound=self.log_sigma_bound,
            est_nll_weight=self.est_nll_weight,
            epochs=self.epochs,
            lr=self.lr,
            seed=self.seed,
            ridge_w=ridge_w,
            ridge_b=ridge_b,
            resid_std=resid_std,
        )
        self._params = DiffusionParams(
            denoiser_layers=_extract_layers(torch, den),
            loc_layers=_extract_layers(torch, loc),
            scale_layers=_extract_layers(torch, scale),
            ctx_mean=np.asarray(ctx_mean, dtype=float),
            ctx_std=np.asarray(ctx_std, dtype=float),
            log_sigma_base=float(log_sigma_base),
            log_sigma_bound=self.log_sigma_bound,
            schedule=sched,
            n_time_freqs=self.n_time_freqs,
            x0_clip=self.x0_clip,
        )
        self.fit_info = DiffPTSFitInfo(
            epochs=self.epochs,
            loss_curve=curve,
            denoise_curve=dcurve,
            est_nll_curve=ncurve,
            final_loss=float(curve[-1]),
            seed=self.seed,
            n_steps=self.n_steps,
        )
        return self

    # -- numpy inference -----------------------------------------------------

    def predict_params(self, X: Array) -> tuple[Array, Array]:
        """LSNM location-scale ``(mu_phi(x), sigma_phi(x))`` per row."""
        return predict_location_scale(self.params, X)

    def predict(self, X: Array) -> Array:
        """Point forecast: the learned location ``mu_phi(x)``."""
        return self.predict_params(X)[0]

    def sample(
        self,
        X: Array,
        n_samples: int = 256,
        *,
        sampler: str = "ddim",
        steps: int = 8,
        seed: int = 0,
        eta: float = 0.0,
    ) -> DiffusionSamples:
        """Predictive draws ``y = mu + sigma z_0`` from the reverse diffusion.

        ``sampler='ddpm'`` runs the full T-step ancestral chain (NFE = T);
        ``'ddim'`` runs the deterministic ``steps``-eval sampler (NFE =
        steps). Proper-score evaluation treats these as an ensemble —
        CRPS is empirical over draws; no closed-form density exists.
        """
        if sampler == "ddpm":
            return ddpm_ancestral_sample(self.params, X, n_samples, seed=seed)
        if sampler == "ddim":
            return ddim_sample(self.params, X, n_samples, steps=steps, seed=seed, eta=eta)
        raise ValueError(f"sampler must be 'ddpm' or 'ddim'; got {sampler!r}")

    def predict_quantiles(
        self,
        X: Array,
        taus: Array,
        *,
        n_samples: int = 256,
        sampler: str = "ddim",
        steps: int = 8,
        seed: int = 0,
    ) -> Array:
        """Sample-quantile estimates, shape (n, len(taus)) — Monte Carlo error applies."""
        t = np.asarray(taus, dtype=float).ravel()
        if (
            t.size == 0
            or not bool(np.all(np.isfinite(t)))
            or bool(np.any(t <= 0.0))
            or bool(np.any(t >= 1.0))
        ):
            raise ValueError("taus must be non-empty and in (0, 1)")
        s = self.sample(X, n_samples, sampler=sampler, steps=steps, seed=seed)
        q = np.quantile(s.samples, t, axis=1).T
        return np.asarray(q, dtype=float)

    def pit(self, X: Array, y: Array, *, n_samples: int = 256, seed: int = 0) -> Array:
        """Randomized PIT against the sample CDF; Uniform(0,1) if calibrated.

        ``u_i ~ U(F_-(y_i), F(y_i))`` with ``F_-``/``F`` the empirical CDF
        bounds of the seeded draws — the same randomized-PIT convention as
        ``models/qrf.py``.
        """
        Xm = _check_matrix(X, "X")
        yv = _check_vector(y, Xm.shape[0], "y")
        s = self.sample(Xm, n_samples, sampler="ddim", seed=seed).samples
        rng = np.random.default_rng(int(seed) + 1)
        below = (s < yv[:, None]).mean(axis=1)
        at = (s <= yv[:, None]).mean(axis=1)
        return np.asarray(below + rng.uniform(size=yv.shape[0]) * (at - below), dtype=float)

    def crps(
        self,
        X: Array,
        y: Array,
        *,
        n_samples: int = 256,
        sampler: str = "ddim",
        steps: int = 8,
        seed: int = 0,
    ) -> float:
        """Mean empirical CRPS of the sampler ensemble (proper score, lower better)."""
        Xm = _check_matrix(X, "X")
        yv = _check_vector(y, Xm.shape[0], "y")
        s = self.sample(Xm, n_samples, sampler=sampler, steps=steps, seed=seed).samples
        vals = [crps_empirical(float(yv[i]), s[i]) for i in range(yv.shape[0])]
        return float(np.mean(np.asarray(vals, dtype=float)))


# ---------------------------------------------------------------------------
# numpy data helpers: windows, synthetic streams, sample scoring
# ---------------------------------------------------------------------------


def make_windows(series: Array, lookback: int) -> tuple[Array, Array]:
    """Trailing-window ``(X, y)`` pairs from a univariate series.

    ``X[i] = series[i : i + lookback]`` (oldest to newest) and
    ``y[i] = series[i + lookback]`` — a single-step-ahead forecasting
    contract. Fail-closed on ``lookback < 1``, a series too short to yield
    one window, or non-finite values.
    """
    s = np.asarray(series, dtype=float).ravel()
    lb = _check_count(lookback, "lookback")
    if s.size <= lb:
        raise ValueError(f"series must have length > lookback={lb}; got {s.size}")
    if not bool(np.all(np.isfinite(s))):
        raise ValueError("series must be finite (NaN/inf rejected)")
    n = s.size - lb
    X = np.lib.stride_tricks.sliding_window_view(s, lb)[:n]
    return np.asarray(X, dtype=float), np.asarray(s[lb:], dtype=float)


def simulate_synthetic_stream(
    n: int, *, kind: str = "ar1_gauss", seed: int = 0, phi: float = 0.7, sigma: float = 1.0
) -> Array:
    """Seeded SYNTHETIC univariate stream — correctness material, never market evidence.

    ``ar1_gauss``: ``y_t = phi y_{t-1} + sigma eps`` (linear-Gaussian, the
    conditional law is Gaussian — a fair baseline check).
    ``ar1_hetero``: innovation scale ``sigma (1 + |y_{t-1}|)``
    (heteroscedastic but symmetric).
    ``ar1_bimodal``: innovation ``sigma (pm w + 0.15 eps)`` with a fair sign
    coin and ``w = 1.2`` — a symmetric two-component residual whose
    conditional law is bimodal, the case where a distributional head should
    differ from a Gaussian baseline. A 64-step burn-in is discarded.
    """
    nn = _check_count(n, "n")
    if kind not in ("ar1_gauss", "ar1_hetero", "ar1_bimodal"):
        raise ValueError(f"kind must be 'ar1_gauss', 'ar1_hetero' or 'ar1_bimodal'; got {kind!r}")
    ar = float(phi)
    if not math.isfinite(ar) or abs(ar) >= 1.0:
        raise ValueError(f"phi must satisfy |phi| < 1; got {phi!r}")
    sg = _check_positive(sigma, "sigma")
    rng = np.random.default_rng(int(seed))
    total = nn + 64
    eps = rng.standard_normal(total)
    y = np.empty(total, dtype=float)
    y[0] = 0.0
    for i in range(1, total):
        if kind == "ar1_hetero":
            y[i] = ar * y[i - 1] + sg * (1.0 + abs(y[i - 1])) * eps[i]
        elif kind == "ar1_bimodal":
            sign = 1.0 if rng.random() < 0.5 else -1.0
            y[i] = ar * y[i - 1] + sg * (sign * 1.2 + 0.15 * eps[i])
        else:
            y[i] = ar * y[i - 1] + sg * eps[i]
    return np.asarray(y[64:], dtype=float)


def _sample_pit(y: Array, samples: Array, rng: np.random.Generator) -> Array:
    below = (samples < y[:, None]).mean(axis=1)
    at = (samples <= y[:, None]).mean(axis=1)
    return np.asarray(below + rng.uniform(size=y.shape[0]) * (at - below), dtype=float)


@dataclass(frozen=True)
class SampleScores:
    """Proper-score summary of a sample ensemble vs observations.

    ``pit_hist`` is the empirical PIT histogram over ``n_bins`` equal bins
    (a calibrated model is uniform); ``pit_ks``/``pit_ks_pvalue`` are the
    Kolmogorov–Smirnov uniformity diagnostic. Everything here is a proper
    score — no Sharpe-family statistic.
    """

    crps: float
    pinball_mean: float
    pinball_by_tau: dict[float, float]
    pit: Array
    pit_hist: Array
    pit_ks: float
    pit_ks_pvalue: float
    coverage_90: float
    mean_width_90: float
    n_obs: int
    n_samples: int


def evaluate_samples(
    y: Array,
    samples: Array,
    taus: Array,
    *,
    n_bins: int = 10,
    seed: int = 0,
) -> SampleScores:
    """Score an ensemble ``samples[i] ~ F(y | x_i)`` against realized ``y``.

    CRPS per row via ``metrics.scoring.crps_empirical``; pinball via
    ``mean_pinball`` on sample quantiles at ``taus``; PIT via the randomized
    empirical-CDF convention, summarized by its histogram and KS statistic.
    Fail-closed on shape mismatches, empty inputs, or invalid taus.
    """
    yv = np.asarray(y, dtype=float).reshape(-1)
    s = np.asarray(samples, dtype=float)
    if s.ndim != 2 or s.shape[0] != yv.shape[0]:
        raise ValueError(f"samples must be (n, M) with n = len(y) = {yv.shape[0]}; got {s.shape}")
    if yv.size == 0:
        raise ValueError("y must be non-empty")
    if s.shape[1] < 1:
        raise ValueError("samples must carry at least one draw per row")
    if not bool(np.all(np.isfinite(yv))) or not bool(np.all(np.isfinite(s))):
        raise ValueError("y and samples must be finite")
    t = np.asarray(taus, dtype=float).ravel()
    if (
        t.size == 0
        or not bool(np.all(np.isfinite(t)))
        or bool(np.any(t <= 0.0))
        or bool(np.any(t >= 1.0))
    ):
        raise ValueError("taus must be non-empty and in (0, 1)")
    nb = _check_count(n_bins, "n_bins")
    q = np.quantile(s, t, axis=1).T
    pin_by_tau = {
        float(tau): mean_pinball(yv, np.asarray(q[:, k]), float(tau)) for k, tau in enumerate(t)
    }
    rng = np.random.default_rng(int(seed))
    pit = _sample_pit(yv, s, rng)
    hist, _ = np.histogram(pit, bins=nb, range=(0.0, 1.0))
    ks, kp = pit_ks(pit)
    q5 = np.quantile(s, 0.05, axis=1)
    q95 = np.quantile(s, 0.95, axis=1)
    inside = (yv >= q5) & (yv <= q95)
    vals = [crps_empirical(float(yv[i]), s[i]) for i in range(yv.shape[0])]
    return SampleScores(
        crps=float(np.mean(np.asarray(vals, dtype=float))),
        pinball_mean=float(np.mean(list(pin_by_tau.values()))),
        pinball_by_tau=pin_by_tau,
        pit=np.asarray(pit, dtype=float),
        pit_hist=np.asarray(hist, dtype=float),
        pit_ks=float(ks),
        pit_ks_pvalue=float(kp),
        coverage_90=float(np.mean(inside)),
        mean_width_90=float(np.mean(q95 - q5)),
        n_obs=int(yv.shape[0]),
        n_samples=int(s.shape[1]),
    )


def evaluate_synthetic_stream(
    n_train: int = 700,
    n_test: int = 250,
    *,
    lookback: int = 16,
    kind: str = "ar1_bimodal",
    seed: int = 0,
    n_samples: int = 192,
    ddim_budgets: Sequence[int] = (2, 4, 8),
    taus: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    hidden: Sequence[int] = (32, 32),
    epochs: int = 150,
    n_steps: int = 50,
) -> dict[str, float | str]:
    """DiffPTS vs NGBoostGaussian and QuantileRegressionForest on a SYNTHETIC stream.

    One seeded stream (``simulate_synthetic_stream``) is windowed and split
    chronologically; every model sees identical train/test rows. DiffPTS is
    scored at each DDIM budget AND at the full DDPM budget — CRPS-vs-NFE
    reporting in the spirit of StocBench (arXiv:2608.22309) — and against the
    proper-score baselines in the spirit of Greenbury et al. (2026,
    arXiv:2606.12997): the ``*_minus_*`` deltas are signed diagnostics, NOT
    a claim that diffusion wins (CRPS-trained ensembles are known to be at
    least as reliable at matched budget). Keys are labeled fixtures:
    ``dgp="fixture"``, ``claim="research_metric_only"``, ``synthetic=kind``;
    proper scores only (AGENTS.md honesty contract). Requires the ``nn``
    extra (torch) for the DiffPTS fit.
    """
    from quant_fund.metrics.scoring import crps_from_quantiles, crps_gaussian
    from quant_fund.models.ngboost_lite import NGBoostGaussian
    from quant_fund.models.qrf import QuantileRegressionForest

    n_tr = _check_count(n_train, "n_train")
    n_te = _check_count(n_test, "n_test")
    lb = _check_count(lookback, "lookback")
    tau_arr = np.asarray(list(taus), dtype=float)
    stream = simulate_synthetic_stream(n_tr + n_te + lb, kind=kind, seed=int(seed))
    X, y = make_windows(stream, lb)
    Xtr, ytr = X[:n_tr], y[:n_tr]
    Xte, yte = X[n_tr : n_tr + n_te], y[n_tr : n_tr + n_te]
    if Xte.shape[0] != n_te:
        raise ValueError("window split produced fewer test rows than requested")

    model = DiffPTSModel(n_steps=n_steps, hidden=hidden, epochs=epochs, seed=int(seed))
    model.fit(Xtr, ytr)
    ngb = NGBoostGaussian(n_estimators=80, learning_rate=0.1, score="crps", seed=int(seed))
    ngb.fit(Xtr, ytr)
    qrf = QuantileRegressionForest(n_estimators=80, min_samples_leaf=5, seed=int(seed))
    qrf.fit(Xtr, ytr)

    out: dict[str, float | str] = {}
    for k in ddim_budgets:
        kk = _check_count(k, "ddim_budget")
        s = model.sample(Xte, n_samples, sampler="ddim", steps=kk, seed=int(seed) + kk)
        sc = evaluate_samples(yte, s.samples, tau_arr, seed=int(seed) + kk)
        out[f"diffpts_ddim{kk}_crps"] = sc.crps
        out[f"diffpts_ddim{kk}_nfe"] = float(s.nfe)
    s_full = model.sample(Xte, n_samples, sampler="ddpm", seed=int(seed) + 999)
    sc_full = evaluate_samples(yte, s_full.samples, tau_arr, seed=int(seed) + 999)
    out["diffpts_ddpm_crps"] = sc_full.crps
    out["diffpts_ddpm_nfe"] = float(s_full.nfe)
    out["diffpts_pinball_mean"] = sc_full.pinball_mean
    out["diffpts_pit_ks"] = sc_full.pit_ks
    out["diffpts_pit_ks_pvalue"] = sc_full.pit_ks_pvalue
    out["diffpts_coverage_90"] = sc_full.coverage_90
    out["diffpts_width_90"] = sc_full.mean_width_90

    mu_ngb, sig_ngb = ngb.predict_params(Xte)
    out["ngboost_crps"] = float(np.mean(crps_gaussian(yte, mu_ngb, sig_ngb)))
    q_ngb = ngb.predict_quantiles(Xte, tau_arr)
    out["ngboost_pinball_mean"] = float(
        np.mean(
            [
                mean_pinball(yte, np.asarray(q_ngb[:, k]), float(tau_arr[k]))
                for k in range(tau_arr.size)
            ]
        )
    )
    ks_ngb, kp_ngb = pit_ks(ngb.pit(Xte, yte))
    out["ngboost_pit_ks"] = float(ks_ngb)
    out["ngboost_pit_ks_pvalue"] = float(kp_ngb)

    q_qrf = qrf.predict_quantiles(Xte, tau_arr)
    out["qrf_pinball_mean"] = float(
        np.mean(
            [
                mean_pinball(yte, np.asarray(q_qrf[:, k]), float(tau_arr[k]))
                for k in range(tau_arr.size)
            ]
        )
    )
    out["qrf_crps"] = float(crps_from_quantiles(yte, np.asarray(q_qrf), tau_arr))
    ks_qrf, kp_qrf = pit_ks(qrf.pit(Xte, yte))
    out["qrf_pit_ks"] = float(ks_qrf)
    out["qrf_pit_ks_pvalue"] = float(kp_qrf)

    ddpm_crps = float(out["diffpts_ddpm_crps"])
    out["crps_diffpts_minus_ngboost"] = ddpm_crps - float(out["ngboost_crps"])
    out["crps_diffpts_minus_qrf"] = ddpm_crps - float(out["qrf_crps"])
    out["n_train"] = float(n_tr)
    out["n_test"] = float(n_te)
    out["lookback"] = float(lb)
    out["n_samples"] = float(n_samples)
    out["n_steps"] = float(n_steps)
    out["seed"] = float(seed)
    out["kind"] = str(kind)
    out["dgp"] = "fixture"
    out["claim"] = "research_metric_only"
    out["synthetic"] = f"SYNTHETIC_{kind}_seeded"
    return out
