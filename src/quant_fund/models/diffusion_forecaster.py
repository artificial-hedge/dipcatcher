"""DiffPTS: diffusion ELBO for probabilistic time-series forecasting.

Ye, W., Li, D., Liu, H., Jiang, H., Sekimoto, Y. & Jiang, R. (2026),
"DiffPTS: Rethinking Diffusion ELBO for Probabilistic Time Series
Forecasting", arXiv:2609.32363 (cs.LG), NeurIPS 2026 Poster, code:
https://github.com/wwy155/DiffPTS. DDPM-based forecasters (TimeGrad, CSDI,
TimeDiff, TMDM, D3U, RDIT, NsDiff) typically train the conditional mean /
variance estimators ``(f_phi, g_psi)`` as separate regression tasks and keep
only the denoising part of the ELBO; DiffPTS rethinks the ELBO under the
Location-Scale Noise Model (LSNM, paper Eq. 8: ``Y_T = f_phi(X) +
sqrt(g_psi(X)) * eta``, ``eta ~ N(0, I)``) and finds that ALL three terms --
reconstruction, denoising matching, and prior matching -- must be optimized
jointly.

Paper math implemented here (verified against arXiv:2609.32363v1, 26 Sep
2026, incl. Appendix A derivations):

- LSNM forward process (Eq. 9): with ``abar_t = prod_{i<=t} (1 - beta_i)``,

      ``Y_t = sqrt(abar_t) Y_0 + (1 - sqrt(abar_t)) f_phi(X)
             + sqrt((1 - abar_t) g_psi(X)) eps_0``,  ``eps_0 ~ N(0, I)``,

  i.e. the diffusion interpolates between the target ``Y_0`` and the learned
  Gaussian endpoint prior ``p(Y_T | X) = N(f_phi(X), g_psi(X))`` (Eq. 15).
- Reverse posterior (Eqs. 10-11): ``q(Y_{t-1} | Y_t, Y_0, X) =
  N(mu_tilde_t, beta_tilde_t g_psi(X))`` with
  ``beta_tilde_t = (1 - abar_{t-1})/(1 - abar_t) * beta_t`` and

      ``mu_tilde_t = c0_t Y_0 + c1_t Y_t + c2_t f_phi(X)``,
      ``c0_t = beta_t sqrt(abar_{t-1}) / (1 - abar_t)``,
      ``c1_t = (1 - abar_{t-1}) sqrt(alpha_t) / (1 - abar_t)``,
      ``c2_t = 1 + (sqrt(abar_t) - 1)(sqrt(alpha_t) + sqrt(abar_{t-1}))
               / (1 - abar_t)``.

  The coefficients satisfy ``c0 + c1 + c2 = 1`` exactly (constant
  trajectories are fixed points), which the tests assert.
- Exact ELBO (Proposition 3.1, Eq. 12):

      ``-ELBO = C + sum_t gamma_t E[||eps_0 - eps_theta(Y_t, X, t)||^2]
               + (abar_T / 2) ||(f_phi(X) - Y_0) / sqrt(g_psi(X))||^2
               + sum_i log[g_psi(X)]_i / 2``,

  ``gamma_t = beta_t / (2 alpha_t (1 - abar_{t-1}))`` for ``t > 1`` and
  ``gamma_1 = 1 / (2 alpha_1)`` -- the prior-matching KL (Appendix A.1,
  Eqs. 28-33) is NOT constant once the endpoint carries parameters, and the
  reconstruction term (Eqs. 24-26) keeps ``log g`` alive. This is the
  "missing puzzle": the ELBO inherently induces a Gaussian NLL objective for
  the estimators and unifies the residual-based (D3U/RDIT) and
  endpoint-based (TMDM/NsDiff) paradigms (Remark 3.2, Appendix A.2).
- Training objective (Proposition 3.3, Eq. 14; the paper's actual objective
  per Appendix E.4 -- "our implementation uses the simplified unweighted
  objective of Proposition 3.3"):

      ``L_ELBO = L_denoise + L_NLL``,
      ``L_denoise = E_{t, eps}[||eps_0 - eps_theta(Y_t, X, t)||^2]``,
      ``L_NLL = (1/2)||(f_phi(X) - Y_0)/sqrt(g_psi(X))||^2
                + sum_i log[g_psi(X)]_i / 2``,

  i.e. joint denoising + conditional-Gaussian NLL, end-to-end in
  ``(theta, phi, psi)`` -- gradients also flow through ``Y_t`` itself
  (Eq. 9 depends on ``phi, psi``). The exact ``gamma_t`` / ``abar_T``
  weighting of Proposition 3.1 is available via ``weighting="elbo"``.
- Sampling (Algorithm 2): ``Y_T ~ N(f_phi(X), g_psi(X))``; for ``t = T..1``
  predict ``eps_theta``, form
  ``Y0_hat = (Y_t - (1 - sqrt(abar_t)) f_phi(X)
              - sqrt((1 - abar_t) g_psi(X)) eps_theta) / sqrt(abar_t)``,
  then step ``Y_{t-1} = mu_tilde_t(Y_t, Y0_hat, X)
                        + sqrt(beta_tilde_t g_psi(X)) z`` for ``t > 1`` and
  ``Y_0 = Y0_hat`` at ``t = 1``.
- Ablations mirror Table 3: ``ablation="no_nll_f"`` (``f_phi == 0``),
  ``"no_nll_g"`` (``g_psi == 1``), ``"no_denoise"`` (drop the denoising
  term; sampling then draws directly from the endpoint ``N(f_phi, g_psi)``).

Deviations from the paper (documented, harness-driven):

1. Univariate marginal on tabular context: one scalar target per
   observation conditioned on feature rows ``X`` (the ``M = D = 1`` case of
   the paper's ``R^{N x D}`` windowed setting) -- the same lane adaptation
   as ``models/odd_residual_flows.py`` (TORF). ``f_phi``/``g_psi`` are small
   MLPs over the context (the paper uses a Non-stationary Transformer for
   ``f_phi`` and a three-layer linear model for ``g_psi``, Appendix B.2);
   ``eps_theta`` is an MLP over the paper's raw view ``[Y_t, one-hot(t),
   X]`` -- the Appendix A.2 residual variable ``Z_t = (Y_t - f)/sqrt(g)``
   (and a ``sqrt(g)`` input feature) were measured to WEAKEN joint
   training: scale-normalized inputs make ``eps`` recoverable for any
   ``g``, removing the denoising gradient through ``Y_t`` that identifies
   ``g_psi`` (the mechanism Proposition 3.3 relies on). One-hot time
   conditioning is exact for the small step counts used here.
2. Noise schedule: COSINE (Nichol, A. & Dhariwal, P. 2021, "Improved
   Denoising Diffusion Probabilistic Models", ICML, PMLR 139:8162-8171) is
   the lane-mandated default; the paper's linear schedule (``beta_1 = 1e-4``
   to ``beta_T = 0.02``, ``T = 20``, giving ``abar_T ~ 0.82``) is available
   via ``schedule="linear"``. Under cosine, ``abar_T -> ~1e-3`` (betas
   clipped at 0.999, Nichol & Dhariwal §4), so the endpoint prior is nearly
   exactly matched by ``q(Y_T | Y_0, X)``.
3. ``Y0_hat`` is clipped to the standardized training-target range plus a
   0.25 margin during sampling -- the standard DDPM stabilization (Ho, J.,
   Jain, A. & Abbeel, P. 2020, NeurIPS 33:6840-6855, §3.2); the paper's
   linear schedule with ``abar_T ~ 0.82`` never needs it, cosine at
   ``abar_T ~ 1e-3`` divides by ``sqrt(abar_T) ~ 0.005`` and does.
4. CRPS: estimated from ``M`` seeded Algorithm-2 samples with the repo's
   ``quant_fund.metrics.scoring.crps_empirical`` (Gneiting, T. & Raftery,
   A.E. 2007, JASA 102:359-378) -- the paper also scores CRPS from 100
   generated samples (Appendix B.3). The empirical estimator is a
   V-statistic with O(1/M) bias and O(1/sqrt(M)) Monte-Carlo noise; all
   comparisons here are seeded and carry documented tolerances. The Gaussian
   NLL endpoint additionally has a CLOSED-FORM CRPS (``crps_gaussian``),
   exposed as :meth:`DiffPTSForecaster.endpoint_crps` with zero MC noise.
5. Optimizer: Adam with cosine LR annealing on CPU single threads (repo
   convention, cf. ``deep_regime_mixture``/``deep_bsde``), small-batch by
   default (``batch_size=250`` with a seeded per-epoch permutation; the
   paper's Appendix B.2 regime is lr 1e-3, batch 32); ``batch_size=None``
   gives full-batch. Deterministic given ``seed`` (GPU determinism not
   claimed).

Honesty (AGENTS.md contract): everything runs on SYNTHETIC seeded streams --
correctness evidence, never market evidence. Scores are proper (CRPS,
endpoint Gaussian log score, PIT, coverage) plus point MAE of the LSNM mean
estimator; no Sharpe/Sortino/Calmar/P&L headline and no live-trading claims.

Conventions: torch is the optional ``nn`` extra, imported lazily via
:func:`_torch` (mirrors ``models/deep_hedging.py`` / TORF), so this module
imports cleanly without torch; only training raises ``ImportError`` with
install guidance. ALL inference (backbone, Algorithm-2 sampling, quantiles,
PIT, CRPS) runs in numpy on float64 parameters extracted after training.
Fail-closed: invalid hyperparameters, non-finite or mismatched inputs,
too-few samples, constant (zero-variance) targets, unfitted use, non-finite
training loss, and non-finite extracted endpoint NLL raise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.scoring import crps_empirical, crps_gaussian, log_score_gaussian

Array = NDArray[np.float64]

__all__ = [
    "DiffPTSFitInfo",
    "DiffPTSForecaster",
    "NoiseSchedule",
    "bench_diffpts",
    "noise_schedule",
]

_SCHEDULES = ("cosine", "linear")
_WEIGHTINGS = ("uniform", "elbo")
_ABLATIONS = ("full", "no_nll_f", "no_nll_g", "no_denoise")
# Cosine-schedule guard rails (Nichol & Dhariwal 2021 §4: clip beta <= 0.999
# so abar_t / abar_{t-1} >= 1e-3 and the 1/sqrt(abar_T) x0-hat inversion in
# Algorithm 2 stays bounded).
_COSINE_OFFSET = 0.008
_BETA_CLIP_MAX = 0.999
_BETA_CLIP_MIN = 1e-8
# Margin (standardized-target units) added around the training target range
# for the Y0_hat clip: DDPM clips to the exact data range; the small slack
# keeps test-set extremes from stacking on the clip boundary.
_Y0_CLIP_MARGIN = 0.25
_MIN_FIT_SAMPLES = 8

_Layers = tuple[tuple[Array, Array], ...]


def _torch() -> Any:
    """Lazily import torch (optional ``nn`` extra); fail closed with guidance."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "DiffPTS needs the optional 'nn' extra (torch): uv sync --extra nn"
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


def _check_choice(value: str, allowed: tuple[str, ...], name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}; got {value!r}")
    return value


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


def _require_finite_loss(value: float, epoch: int) -> float:
    """Fail-closed guard on the training loss (mirrors deep_bsde/deep_regime)."""
    if not math.isfinite(value):
        raise ValueError(
            f"DiffPTS training loss is not finite at epoch {epoch} (loss={value!r}); "
            "reduce lr or check inputs"
        )
    return value


# ---------------------------------------------------------------------------
# noise schedule (numpy, deterministic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class NoiseSchedule:
    """Diffusion schedule arrays, 1-based step ``t = 1..T`` at index ``t-1``.

    ``gamma`` holds the exact-ELBO denoising weights of Proposition 3.1
    (Eq. 12); ``c0``/``c1``/``c2`` the posterior-mean coefficients of Eq. 11
    (``c0 + c1 + c2 = 1`` exactly); ``beta_tilde[0] = 0`` because the
    ``t = 1`` reverse branch returns ``Y0_hat`` directly (Algorithm 2).
    """

    kind: str
    n_steps: int
    beta: Array
    alpha: Array
    alpha_bar: Array
    sqrt_alpha_bar: Array
    sqrt_beta_bar: Array  # sqrt(1 - alpha_bar), the paper's sqrt(beta_bar_t)
    beta_tilde: Array
    gamma: Array
    c0: Array
    c1: Array
    c2: Array

    @property
    def alpha_bar_T(self) -> float:
        return float(self.alpha_bar[-1])


def noise_schedule(
    n_steps: int,
    kind: str = "cosine",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> NoiseSchedule:
    """Build the forward-process schedule (all arrays float64, deterministic).

    ``kind="cosine"``: Nichol & Dhariwal (2021) -- ``abar_t ~ f(t)/f(0)`` with
    ``f(t) = cos^2((t/T + s)/(1 + s) * pi/2)``, ``s = 0.008``, betas clipped
    to ``[1e-8, 0.999]`` and ``abar`` recomputed from the clipped betas so
    every downstream identity holds for the schedule actually used.
    ``kind="linear"``: the paper's setting (betas linear from ``beta_start``
    to ``beta_end``; Appendix E.4 quotes ``abar_T ~ 0.82`` at ``T = 20``).
    """
    T = _check_count(n_steps, "n_steps")
    _check_choice(kind, _SCHEDULES, "schedule")
    if kind == "linear":
        b0 = _check_positive(beta_start, "beta_start")
        b1 = _check_positive(beta_end, "beta_end")
        if not (b0 < b1 < 1.0):
            raise ValueError(
                f"need 0 < beta_start < beta_end < 1; got ({beta_start!r}, {beta_end!r})"
            )
        beta = np.linspace(b0, b1, T)
    else:
        s = _COSINE_OFFSET
        steps = np.arange(1, T + 1, dtype=float)
        f = np.cos(((steps / T + s) / (1.0 + s)) * math.pi / 2.0) ** 2
        f0 = math.cos((s / (1.0 + s)) * math.pi / 2.0) ** 2
        abar_raw = f / f0
        beta = 1.0 - abar_raw / np.concatenate([[1.0], abar_raw[:-1]])
        beta = np.clip(beta, _BETA_CLIP_MIN, _BETA_CLIP_MAX)
    alpha = 1.0 - beta
    abar = np.cumprod(alpha)
    abar_prev = np.concatenate([[1.0], abar[:-1]])  # abar_0 := 1
    one_m_abar = 1.0 - abar
    # beta_tilde_t = (1 - abar_{t-1}) / (1 - abar_t) * beta_t; 0 at t = 1.
    beta_tilde = (1.0 - abar_prev) / one_m_abar * beta
    beta_tilde[0] = 0.0
    # gamma_t: exact-ELBO denoising weights (Prop 3.1); gamma_1 = 1/(2 abar_1)
    # equals the reconstruction weight 1/(2 alpha_1) since abar_1 = alpha_1
    # (abar_0 = 1 would divide by zero in the general formula, so t = 1 is set
    # from its own closed form).
    gamma = np.empty(T, dtype=float)
    gamma[1:] = beta[1:] / (2.0 * alpha[1:] * (1.0 - abar_prev[1:]))
    gamma[0] = 1.0 / (2.0 * alpha[0])
    sqrt_abar = np.sqrt(abar)
    sqrt_abar_prev = np.sqrt(abar_prev)
    sqrt_alpha = np.sqrt(alpha)
    c0 = beta * sqrt_abar_prev / one_m_abar
    c1 = (1.0 - abar_prev) * sqrt_alpha / one_m_abar
    c2 = 1.0 + (sqrt_abar - 1.0) * (sqrt_alpha + sqrt_abar_prev) / one_m_abar
    return NoiseSchedule(
        kind=str(kind),
        n_steps=T,
        beta=np.asarray(beta, dtype=float),
        alpha=np.asarray(alpha, dtype=float),
        alpha_bar=np.asarray(abar, dtype=float),
        sqrt_alpha_bar=np.asarray(sqrt_abar, dtype=float),
        sqrt_beta_bar=np.asarray(np.sqrt(one_m_abar), dtype=float),
        beta_tilde=np.asarray(beta_tilde, dtype=float),
        gamma=np.asarray(gamma, dtype=float),
        c0=np.asarray(c0, dtype=float),
        c1=np.asarray(c1, dtype=float),
        c2=np.asarray(c2, dtype=float),
    )


# ---------------------------------------------------------------------------
# numpy inference core (pure numpy on extracted float64 parameters)
# ---------------------------------------------------------------------------


def _np_mlp(x: Array, layers: Sequence[tuple[Array, Array]]) -> Array:
    """Numpy mirror of the training MLPs: tanh hidden layers, linear output."""
    z = x
    last = len(layers) - 1
    for i, (W, b) in enumerate(layers):
        z = z @ W.T + b
        if i < last:
            z = np.tanh(z)
    return np.asarray(z, dtype=float)


@dataclass(frozen=True, eq=False)
class _NumpyDiffPTSParams:
    """Float64 inference parameters extracted from the torch training run."""

    schedule: NoiseSchedule
    ablation: str
    log_g_clip: float
    y_mean: float
    y_std: float
    ctx_mean: Array
    ctx_std: Array
    f_layers: _Layers | None
    g_layers: _Layers | None
    eps_layers: _Layers | None
    y0_clip_lo: float
    y0_clip_hi: float


def _np_backbone(params: _NumpyDiffPTSParams, Xs: Array) -> tuple[Array, Array]:
    """LSNM estimators in STANDARDIZED target units: mean ``f``, variance ``g``.

    ``f = 0`` when the mean estimator is ablated (``no_nll_f``); ``g = 1``
    when the variance estimator is ablated (``no_nll_g``). ``log g`` is
    produced by a linear head clipped to ``[-c, c]`` (repo log-scale clip
    convention, cf. ``ngboost_lite._LOG_SIGMA_CLIP``), so ``g`` is strictly
    positive and bounded away from 0/inf -- the NLL's ``log g`` term keeps
    the pair ``(quadratic, log)`` jointly minimized at the conditional
    residual variance.
    """
    n = Xs.shape[0]
    if params.f_layers is None:
        f = np.zeros(n)
    else:
        f = _np_mlp(Xs, params.f_layers)[..., 0]
    if params.g_layers is None:
        g = np.ones(n)
    else:
        log_g = np.clip(_np_mlp(Xs, params.g_layers)[..., 0], -params.log_g_clip, params.log_g_clip)
        g = np.exp(log_g)
    return np.asarray(f, dtype=float), np.asarray(g, dtype=float)


def _np_eps(params: _NumpyDiffPTSParams, Y: Array, Xs: Array, t: int) -> Array:
    """eps_theta(Y_t, X, t) with one-hot time conditioning; (n, ...) -> (n, ...).

    The paper's raw ``Y_t`` view (NOT the Appendix A.2 residual variable
    ``Z_t = (Y_t - f)/sqrt(g)``, and NOT augmented with ``sqrt(g)``): both
    scale-normalized variants were measured to WEAKEN joint training --
    they let ``eps`` become recoverable for (nearly) any ``g``, removing
    the denoising gradient through ``Y_t`` that identifies ``g_psi`` --
    which is exactly the mechanism Proposition 3.3 relies on.
    """
    if params.eps_layers is None:
        raise RuntimeError("eps network is not available (ablation without denoising)")
    sched = params.schedule
    if not 1 <= t <= sched.n_steps:
        raise ValueError(f"t must be in 1..{sched.n_steps}; got {t}")
    shape = Y.shape
    extra = shape[1:]
    p = Xs.shape[1]
    onehot = np.broadcast_to(np.eye(sched.n_steps, dtype=float)[t - 1], shape + (sched.n_steps,))
    Xb = np.broadcast_to(Xs.reshape((Xs.shape[0],) + (1,) * len(extra) + (p,)), shape + (p,))
    inp = np.concatenate([Y[..., None], onehot, Xb], axis=-1).reshape(-1, 1 + sched.n_steps + p)
    return np.asarray(_np_mlp(inp, params.eps_layers).reshape(shape), dtype=float)


def _np_sample(params: _NumpyDiffPTSParams, Xs: Array, n_samples: int, seed: int) -> Array:
    """Algorithm 2 (sampling), vectorized over observations and draws.

    Returns standardized-unit targets, shape (n, M). Deterministic given
    ``seed``: one ``default_rng`` draws the endpoint noise ``eta`` first,
    then the reverse-step noises ``z`` for ``t = T..2`` in order. The
    ``no_denoise`` ablation samples the endpoint ``N(f, g)`` directly (the
    paper's Table 3 evaluation protocol for that variant).
    """
    n = Xs.shape[0]
    f, g = _np_backbone(params, Xs)
    rng = np.random.default_rng(int(seed))
    eta = rng.standard_normal((n, n_samples))
    sqrt_g = np.sqrt(g)[:, None]
    if params.eps_layers is None:
        return np.asarray(f[:, None] + sqrt_g * eta, dtype=float)
    sched = params.schedule
    fc = f[:, None]
    sqrt_g = np.broadcast_to(sqrt_g, (n, n_samples))
    Y = fc + sqrt_g * eta  # Y_T ~ N(f, g)   (Eq. 15 prior)
    for t in range(sched.n_steps, 0, -1):
        eps_hat = _np_eps(params, Y, Xs, t)
        sab = sched.sqrt_alpha_bar[t - 1]
        sbb = sched.sqrt_beta_bar[t - 1]
        # x0-prediction (Algorithm 2 step 5 / Eq. 20-21 rearranged)
        Y0 = (Y - (1.0 - sab) * fc - sbb * sqrt_g * eps_hat) / sab
        Y0 = np.clip(Y0, params.y0_clip_lo, params.y0_clip_hi)
        if t == 1:
            Y = Y0
        else:
            z = rng.standard_normal((n, n_samples))
            mu = sched.c0[t - 1] * Y0 + sched.c1[t - 1] * Y + sched.c2[t - 1] * fc
            Y = mu + math.sqrt(float(sched.beta_tilde[t - 1])) * sqrt_g * z
    return np.asarray(Y, dtype=float)


# ---------------------------------------------------------------------------
# torch training core (guarded; the numpy inference mirrors it exactly)
# ---------------------------------------------------------------------------


def _build_mlp(torch: Any, d_in: int, hidden: Sequence[int], d_out: int) -> Any:
    layers: list[Any] = []
    d = int(d_in)
    for hh in hidden:
        layers += [torch.nn.Linear(d, int(hh)), torch.nn.Tanh()]
        d = int(hh)
    layers.append(torch.nn.Linear(d, int(d_out)))
    return torch.nn.Sequential(*layers)


def _init_zero_last(torch: Any, net: Any) -> None:
    """Zero the output layer: f starts at 0 and log g at 0 (g = 1), so the
    endpoint prior STARTS as the standardized marginal N(0, 1) and the
    forward process (Eq. 9) starts as a vanilla DDPM on standardized y."""
    last = net[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()


def _extract_mlp_layers(torch: Any, net: Any) -> _Layers:
    out: list[tuple[Array, Array]] = []
    for module in net:
        if isinstance(module, torch.nn.Linear):
            W = np.array(module.weight.detach().numpy(), dtype=float)
            b = np.array(module.bias.detach().numpy(), dtype=float)
            out.append((W, b))
    return tuple(out)


def _train_diffpts(
    torch: Any,
    Xs: Array,
    ys: Array,
    *,
    sched: NoiseSchedule,
    hidden: Sequence[int],
    log_g_clip: float,
    epochs: int,
    lr: float,
    weighting: str,
    ablation: str,
    batch_size: int | None,
    seed: int,
) -> tuple[Any, Any, Any, list[float], list[float], list[float]]:
    """Adam on the ELBO-derived joint objective; deterministic given ``seed``.

    Full-batch by default; ``batch_size`` enables the paper's mini-batch
    regime (Appendix B.2: batch 32, Adam lr 1e-3) with a seeded per-epoch
    permutation, so every update stays reproducible. Per update: draw
    ``t ~ U{1..T}`` per observation and ``eps_0 ~ N(0, I)`` from a seeded
    numpy generator (independent of the torch RNG stream, so initialization
    and training noise never interleave), form ``Y_t`` by Eq. 9, and
    minimize ``L_denoise + L_NLL`` (Eq. 14) -- or the exact ``gamma_t`` /
    ``abar_T``-weighted Proposition 3.1 objective when ``weighting="elbo"``.
    Gradients flow to ``(theta, phi, psi)`` jointly: ``Y_t`` itself depends
    on ``f_phi``/``g_psi`` (the paper's central point -- the denoising term
    trains the estimators too). Cosine LR annealing mirrors
    ``deep_regime_mixture``; curves are recorded per UPDATE (per epoch when
    full-batch).
    """
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    n, p = int(Xs.shape[0]), int(Xs.shape[1])
    T = int(sched.n_steps)
    f_net = None if ablation == "no_nll_f" else _build_mlp(torch, p, hidden, 1)
    g_net = None if ablation == "no_nll_g" else _build_mlp(torch, p, hidden, 1)
    # eps input: [Y_t, one-hot(t), Xs] (see _np_eps).
    eps_net = None if ablation == "no_denoise" else _build_mlp(torch, 1 + T + p, hidden, 1)
    for net in (f_net, g_net):
        if net is not None:
            _init_zero_last(torch, net)
    params: list[Any] = []
    for net in (f_net, g_net, eps_net):
        if net is not None:
            params.extend(net.parameters())
    opt = torch.optim.Adam(params, lr=float(lr))
    bsz = n if batch_size is None else min(int(batch_size), n)
    steps_per_epoch = max(1, -(-n // bsz))  # ceil division
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(epochs) * steps_per_epoch, eta_min=lr / 10.0
    )
    X_all = torch.as_tensor(Xs, dtype=torch.float32)
    y_all = torch.as_tensor(ys, dtype=torch.float32)
    sab = torch.as_tensor(sched.sqrt_alpha_bar, dtype=torch.float32)
    sbb = torch.as_tensor(sched.sqrt_beta_bar, dtype=torch.float32)
    gam = torch.as_tensor(sched.gamma, dtype=torch.float32)
    quad_w = float(sched.alpha_bar_T) if weighting == "elbo" else 1.0
    rng = np.random.default_rng(int(seed))
    loss_curve: list[float] = []
    denoise_curve: list[float] = []
    nll_curve: list[float] = []
    update = 0
    for _epoch in range(int(epochs)):
        order = np.arange(n) if batch_size is None else rng.permutation(n)
        for start in range(0, n, bsz):
            idx = order[start : start + bsz]
            X_t = X_all if batch_size is None else X_all[idx]
            y_t = y_all if batch_size is None else y_all[idx]
            nb = int(idx.size)
            t_idx = torch.as_tensor(rng.integers(0, T, size=nb), dtype=torch.int64)
            eps0 = torch.as_tensor(rng.standard_normal(nb), dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            if f_net is None:
                f = torch.zeros_like(y_t)
            else:
                f = f_net(X_t).squeeze(-1)
            if g_net is None:
                log_g = torch.zeros_like(y_t)
            else:
                log_g = torch.clamp(g_net(X_t).squeeze(-1), -float(log_g_clip), float(log_g_clip))
            g = torch.exp(log_g)
            sqrt_g = torch.sqrt(g)
            sqrt_abar = sab[t_idx]
            Y = sqrt_abar * y_t + (1.0 - sqrt_abar) * f + sbb[t_idx] * sqrt_g * eps0
            if eps_net is None:
                denoise = torch.zeros((), dtype=torch.float32)
            else:
                onehot = torch.nn.functional.one_hot(t_idx, num_classes=T).to(torch.float32)
                eps_hat = eps_net(torch.cat([Y.unsqueeze(-1), onehot, X_t], dim=-1)).squeeze(-1)
                resid_sq = (eps0 - eps_hat) * (eps0 - eps_hat)
                if weighting == "elbo":
                    denoise = torch.mean(gam[t_idx] * resid_sq)
                else:
                    denoise = torch.mean(resid_sq)
            # Gaussian NLL for the endpoint (Eq. 14 right term / Eq. 12 tail):
            # (1/2) (f - y)^2 / g + (1/2) log g, the quadratic weighted by
            # abar_T under the exact Proposition 3.1 weighting.
            nll = quad_w * 0.5 * torch.mean((f - y_t) * (f - y_t) / g) + 0.5 * torch.mean(log_g)
            loss = denoise + nll
            loss_curve.append(_require_finite_loss(float(loss.detach().numpy()), update))
            denoise_curve.append(float(denoise.detach().numpy()))
            nll_curve.append(float(nll.detach().numpy()))
            loss.backward()
            opt.step()
            lr_sched.step()
            update += 1
    return f_net, g_net, eps_net, loss_curve, denoise_curve, nll_curve


# ---------------------------------------------------------------------------
# the forecaster
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffPTSFitInfo:
    """Training trace. Curves are recorded BEFORE each Adam update (entry 0
    is the loss at the zero-init endpoint ``N(0, 1)``); ``denoise``/``nll``
    split the joint objective (Eq. 14). ``final_endpoint_nll`` is the
    Gaussian NLL of the RAW targets under the extracted numpy endpoint
    ``N(f, g)`` recomputed on the training data (includes the ``log y_std``
    change-of-variables term)."""

    epochs: int
    n_steps: int
    schedule: str
    weighting: str
    ablation: str
    seed: int
    loss_curve: list[float]
    denoise_curve: list[float]
    nll_curve: list[float]
    final_endpoint_nll: float


class DiffPTSForecaster:
    """DiffPTS forecaster (arXiv:2609.32363): LSNM endpoint + DDPM refinement.

    ``fit`` jointly trains ``(eps_theta, f_phi, g_psi)`` on the ELBO-derived
    objective (Eq. 14); the predictive distribution is generated by the
    Algorithm-2 reverse pass from the learned endpoint ``N(f_phi(X),
    g_psi(X))``. The point forecast (:meth:`predict`) is the LSNM conditional
    mean ``f_phi(X)`` -- unlike TORF it is NOT frozen against the density
    refinement (the paper's trade: end-to-end alignment instead of exact mean
    preservation). ``fit`` needs torch (the ``nn`` extra); every inference
    method is pure numpy on the extracted float64 parameters.
    """

    def __init__(
        self,
        n_steps: int = 20,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        hidden: Sequence[int] = (64, 64),
        epochs: int = 250,
        lr: float = 3e-3,
        weighting: str = "uniform",
        ablation: str = "full",
        batch_size: int | None = 250,
        log_g_clip: float = 12.0,
        seed: int = 0,
    ) -> None:
        if isinstance(n_steps, bool) or int(n_steps) != n_steps or int(n_steps) < 1:
            raise ValueError(f"n_steps must be an int >= 1; got {n_steps!r}")
        if batch_size is not None and (
            isinstance(batch_size, bool) or int(batch_size) != batch_size or int(batch_size) < 1
        ):
            raise ValueError(f"batch_size must be an int >= 1 or None; got {batch_size!r}")
        _check_choice(schedule, _SCHEDULES, "schedule")
        _check_choice(weighting, _WEIGHTINGS, "weighting")
        _check_choice(ablation, _ABLATIONS, "ablation")
        hidden_widths = tuple(int(hh) for hh in hidden)
        if not hidden_widths or any(hh < 1 for hh in hidden_widths):
            raise ValueError(
                f"hidden must be a non-empty sequence of positive widths; got {hidden!r}"
            )
        if isinstance(epochs, bool) or int(epochs) != epochs or int(epochs) < 1:
            raise ValueError(f"epochs must be an int >= 1; got {epochs!r}")
        self.n_steps = int(n_steps)
        self.schedule = str(schedule)
        self.beta_start = _check_positive(beta_start, "beta_start")
        self.beta_end = _check_positive(beta_end, "beta_end")
        self.hidden = hidden_widths
        self.epochs = int(epochs)
        self.lr = _check_positive(lr, "lr")
        self.weighting = str(weighting)
        self.ablation = str(ablation)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.log_g_clip = _check_positive(log_g_clip, "log_g_clip")
        self.seed = int(seed)
        self._params: _NumpyDiffPTSParams | None = None
        self.fit_info: DiffPTSFitInfo | None = None

    # -- state ---------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._params is not None

    @property
    def n_features(self) -> int:
        return int(self._require_params().ctx_mean.size)

    def _require_params(self) -> _NumpyDiffPTSParams:
        if self._params is None:
            raise RuntimeError("DiffPTSForecaster is not fitted")
        return self._params

    def _context(self, C: Array) -> Array:
        params = self._require_params()
        arr = _check_matrix(C, "X")
        if arr.shape[1] != params.ctx_mean.size:
            raise ValueError(
                f"feature count mismatch: fitted with {params.ctx_mean.size}, got {arr.shape[1]}"
            )
        return arr

    def _standardize(self, C: Array) -> Array:
        params = self._require_params()
        return np.asarray((C - params.ctx_mean) / params.ctx_std, dtype=float)

    # -- training (torch-gated) ------------------------------------------------

    def fit(self, X: Array, y: Array) -> DiffPTSForecaster:
        """Fit DiffPTS end-to-end on the joint ELBO objective (Eq. 14).

        Inputs are validated BEFORE torch is touched, so input-contract
        errors raise ``ValueError`` even without the ``nn`` extra; a
        successful fit needs torch and raises ``ImportError`` with guidance
        when absent.
        """
        Xm = _check_matrix(X, "X")
        yv = _check_vector(y, Xm.shape[0], "y")
        if Xm.shape[0] < _MIN_FIT_SAMPLES:
            raise ValueError(f"too few samples: need >= {_MIN_FIT_SAMPLES}, got {Xm.shape[0]}")
        y_std = float(np.std(yv))
        if not math.isfinite(y_std) or y_std <= 0.0:
            raise ValueError("y has zero variance (constant series); nothing to fit")
        y_mean = float(np.mean(yv))
        ctx_mean = Xm.mean(axis=0)
        ctx_std = Xm.std(axis=0)
        ctx_std = np.where(ctx_std > 0.0, ctx_std, 1.0)  # constant columns: unit scale
        Xs = np.asarray((Xm - ctx_mean) / ctx_std, dtype=float)
        ys = np.asarray((yv - y_mean) / y_std, dtype=float)
        sched = noise_schedule(self.n_steps, self.schedule, self.beta_start, self.beta_end)

        torch = _torch()
        f_net, g_net, eps_net, loss_curve, denoise_curve, nll_curve = _train_diffpts(
            torch,
            Xs,
            ys,
            sched=sched,
            hidden=self.hidden,
            log_g_clip=self.log_g_clip,
            epochs=self.epochs,
            lr=self.lr,
            weighting=self.weighting,
            ablation=self.ablation,
            batch_size=self.batch_size,
            seed=self.seed,
        )
        ys_lo = float(ys.min()) - _Y0_CLIP_MARGIN
        ys_hi = float(ys.max()) + _Y0_CLIP_MARGIN
        params = _NumpyDiffPTSParams(
            schedule=sched,
            ablation=self.ablation,
            log_g_clip=self.log_g_clip,
            y_mean=y_mean,
            y_std=y_std,
            ctx_mean=np.asarray(ctx_mean, dtype=float),
            ctx_std=np.asarray(ctx_std, dtype=float),
            f_layers=None if f_net is None else _extract_mlp_layers(torch, f_net),
            g_layers=None if g_net is None else _extract_mlp_layers(torch, g_net),
            eps_layers=None if eps_net is None else _extract_mlp_layers(torch, eps_net),
            y0_clip_lo=ys_lo,
            y0_clip_hi=ys_hi,
        )
        self._params = params
        f_raw, sigma_raw = self.predict_params(Xm)
        nll = 0.5 * np.mean(((f_raw - yv) / sigma_raw) ** 2 + np.log(2.0 * math.pi * sigma_raw**2))
        if not math.isfinite(nll):
            self._params = None
            raise ValueError("extracted endpoint parameters produce a non-finite NLL")
        self.fit_info = DiffPTSFitInfo(
            epochs=self.epochs,
            n_steps=self.n_steps,
            schedule=self.schedule,
            weighting=self.weighting,
            ablation=self.ablation,
            seed=self.seed,
            loss_curve=loss_curve,
            denoise_curve=denoise_curve,
            nll_curve=nll_curve,
            final_endpoint_nll=float(nll),
        )
        return self

    # -- numpy inference -------------------------------------------------------

    def predict_params(self, X: Array) -> tuple[Array, Array]:
        """Endpoint Gaussian parameters in RAW units: ``(f_phi(X), sigma)``.

        ``sigma = y_std * sqrt(g_psi)`` with the LSNM variance ``g_psi`` in
        standardized units; strictly positive by the clipped log-g head.
        """
        params = self._require_params()
        f, g = _np_backbone(params, self._standardize(self._context(X)))
        f_raw = np.asarray(params.y_mean + params.y_std * f, dtype=float)
        sigma_raw = np.asarray(params.y_std * np.sqrt(g), dtype=float)
        if not (bool(np.all(np.isfinite(f_raw))) and bool(np.all(np.isfinite(sigma_raw)))):
            raise ValueError("endpoint parameters are not finite")
        if bool(np.any(sigma_raw <= 0.0)):
            raise ValueError("endpoint sigma must be positive")
        return f_raw, sigma_raw

    def predict(self, X: Array) -> Array:
        """Point forecast: the LSNM conditional-mean estimator ``f_phi(X)``."""
        return self.predict_params(X)[0]

    def predict_sigma(self, X: Array) -> Array:
        """Endpoint standard deviation ``sqrt(g_psi(X))`` in raw units."""
        return self.predict_params(X)[1]

    def predict_samples(self, X: Array, n_samples: int, seed: int = 0) -> Array:
        """Seeded Algorithm-2 draws from ``p(Y_0 | X)``, shape (n, M), raw units."""
        params = self._require_params()
        Xm = self._context(X)
        m = _check_count(n_samples, "n_samples")
        Xs = self._standardize(Xm)
        y_std = _np_sample(params, Xs, m, int(seed))
        return np.asarray(params.y_mean + params.y_std * y_std, dtype=float)

    def predict_quantiles(
        self, X: Array, taus: Array, n_samples: int = 400, seed: int = 0
    ) -> Array:
        """Predictive quantiles from seeded samples, shape (n, T).

        Sample quantiles (linear interpolation) are monotone in ``tau`` by
        construction, so columns never cross for sorted ``taus``.
        """
        t = np.asarray(taus, dtype=float).ravel()
        if t.size == 0 or not bool(np.all(np.isfinite(t))) or np.any(t <= 0.0) or np.any(t >= 1.0):
            raise ValueError(f"taus must be non-empty, finite, and in (0, 1); got {taus!r}")
        samples = self.predict_samples(X, n_samples, seed=seed)
        return np.asarray(np.quantile(samples, t, axis=1).T, dtype=float)

    def pit(self, X: Array, y: Array, n_samples: int = 400, seed: int = 0) -> Array:
        """Randomized PIT via the seeded empirical CDF (midrank + half-step):

        ``F_hat(y) = (#{s < y} + 0.5 #{s = y} + 0.5) / (M + 1)``, strictly
        inside (0, 1); uniform under a calibrated predictive. Discreteness of
        the M-sample CDF bounds PIT resolution at ~1/(M+1) -- documented MC
        tolerance for KS-based calibration checks.
        """
        Xm = self._context(X)
        yv = _check_vector(y, Xm.shape[0], "y")
        samples = self.predict_samples(Xm, n_samples, seed=seed)
        less = np.sum(samples < yv[:, None], axis=1)
        equal = np.sum(samples == yv[:, None], axis=1)
        return np.asarray((less + 0.5 * equal + 0.5) / (n_samples + 1.0), dtype=float)

    def crps(self, X: Array, y: Array, n_samples: int = 400, seed: int = 0) -> float:
        """Mean CRPS of the FULL diffusion predictive from seeded samples.

        Uses ``quant_fund.metrics.scoring.crps_empirical`` per observation
        (Gneiting & Raftery 2007 ensemble form; the paper also scores CRPS
        from generated samples, Appendix B.3). Monte-Carlo error is
        O(1/sqrt(M)) with an O(1/M) V-statistic bias; comparisons in tests
        and benches are seeded and carry documented tolerances. For the
        zero-MC-noise score of the Gaussian endpoint component see
        :meth:`endpoint_crps`.
        """
        Xm = self._context(X)
        yv = _check_vector(y, Xm.shape[0], "y")
        samples = self.predict_samples(Xm, n_samples, seed=seed)
        scores = [crps_empirical(float(yv[i]), samples[i]) for i in range(yv.size)]
        out = float(np.mean(np.asarray(scores, dtype=float)))
        if not math.isfinite(out):
            raise ValueError("sample CRPS is not finite")
        return out

    def endpoint_crps(self, X: Array, y: Array) -> float:
        """CLOSED-FORM mean CRPS of the endpoint Gaussian ``N(f_phi, g_psi)``.

        The LSNM/NLL component of the ELBO scored alone (the paper's "w/o
        denoise" evaluation samples exactly this distribution) -- no
        sampling noise.
        """
        Xm = self._context(X)
        yv = _check_vector(y, Xm.shape[0], "y")
        f, sigma = self.predict_params(Xm)
        return float(np.mean(crps_gaussian(yv, f, sigma)))

    def endpoint_log_score(self, X: Array, y: Array) -> float:
        """Mean Gaussian log score of the endpoint ``N(f_phi, g_psi)`` (higher better)."""
        Xm = self._context(X)
        yv = _check_vector(y, Xm.shape[0], "y")
        f, sigma = self.predict_params(Xm)
        return float(np.mean(log_score_gaussian(yv, f, sigma)))

    def mae(self, X: Array, y: Array) -> float:
        """Point MAE of the LSNM mean estimator ``f_phi``."""
        mu = self.predict(X)
        yv = _check_vector(y, X.shape[0], "y")
        return float(np.mean(np.abs(yv - mu)))


# ---------------------------------------------------------------------------
# SYNTHETIC validation bench
# ---------------------------------------------------------------------------


def _synthetic_hetero_stream(
    n_train: int, n_test: int, seed: int, noise: str
) -> tuple[Array, Array, Array, Array]:
    """SYNTHETIC seeded heteroskedastic stream -- correctness, never market evidence.

    Composed to be BITWISE IDENTICAL to the TORF lane's stream
    (``odd_residual_flows._synthetic_hetero_stream``, itself adapting
    arXiv:2608.11114 Appendix C): same DGP, same rng call order, so the same
    seed yields the same data and the DiffPTS/TORF/NGBoost comparison runs
    on shared streams. Mean ``mu = 2 x0 + x1^2`` (linear in the provided
    features), heteroskedastic scale ``gamma(x) = 0.3 + 1.2 (x1 + 1)/2``;
    ``noise='student_t'``: symmetric heavy tails (nu = 4, variance-normalized
    by gamma); ``noise='bimodal'``: symmetric two-point residual of width
    gamma with slight jitter. Both violate the Gaussian assumption of
    NGBoost-style baselines while keeping ``E[y | x] = mu(x)`` exact.
    """
    if noise not in ("student_t", "bimodal"):
        raise ValueError(f"noise must be 'student_t' or 'bimodal'; got {noise!r}")
    n_tr = _check_count(n_train, "n_train")
    n_te = _check_count(n_test, "n_test")
    rng = np.random.default_rng(int(seed))
    n = n_tr + n_te
    x0 = rng.uniform(-1.0, 1.0, size=n)
    x1 = rng.uniform(-1.0, 1.0, size=n)
    mu = 2.0 * x0 + x1 * x1
    feats = np.column_stack([x0, x1, x1 * x1])
    gamma = 0.3 + 1.2 * (x1 + 1.0) / 2.0
    if noise == "student_t":
        nu = 4.0
        eps = gamma * rng.standard_t(nu, size=n) / math.sqrt(nu / (nu - 2.0))
    else:
        sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
        eps = gamma * (sign + 0.15 * rng.standard_normal(n))
    y = mu + eps
    return (
        np.asarray(feats[:n_tr], dtype=float),
        np.asarray(y[:n_tr], dtype=float),
        np.asarray(feats[n_tr:], dtype=float),
        np.asarray(y[n_tr:], dtype=float),
    )


def bench_diffpts(
    n_train: int = 1000,
    n_test: int = 600,
    seed: int = 0,
    noise: str = "student_t",
    n_steps: int = 20,
    schedule: str = "cosine",
    epochs: int = 250,
    lr: float = 3e-3,
    batch_size: int | None = 250,
    hidden: Sequence[int] = (64, 64),
    n_samples: int = 400,
    ngboost_rounds: int = 80,
    ngboost_lr: float = 0.1,
    torf_epochs: int = 250,
) -> dict[str, float | str]:
    """DiffPTS vs NGBoostGaussian vs TORF on a SYNTHETIC seeded stream.

    Correctness bench for the paper's central claim (arXiv:2609.32363 §4.2,
    Table 2): the full-ELBO diffusion beats the Gaussian density baseline on
    CRPS -- here on the SHARED heteroskedastic streams of the TORF/DeRegiME
    lanes (bitwise-identical fixture, same seeds), against
    ``NGBoostGaussian(score='crps')`` (Duan et al. 2020) and the TORF odd
    residual flow (arXiv:2608.11114). ``diffpts_crps`` is the sample-based
    proper score (documented MC tolerance); ``diffpts_endpoint_crps`` is the
    closed-form CRPS of the LSNM Gaussian endpoint alone (the paper's "w/o
    denoise" ablation level). All models see identical features and data;
    everything is seeded. Proper scores plus point MAE only; labeled
    ``dgp=fixture`` / ``claim=research_metric_only``; never market evidence,
    no Sharpe/P&L (AGENTS.md honesty contract). Requires the torch ``nn``
    extra for the DiffPTS/TORF fits.
    """
    from quant_fund.metrics.probability import pit_ks
    from quant_fund.models.ngboost_lite import NGBoostGaussian
    from quant_fund.models.odd_residual_flows import OddResidualFlow, TORFForecaster

    Xtr, ytr, Xte, yte = _synthetic_hetero_stream(n_train, n_test, seed, noise)
    diffpts = DiffPTSForecaster(
        n_steps=n_steps,
        schedule=schedule,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        hidden=hidden,
        seed=int(seed),
    ).fit(Xtr, ytr)
    ngb = NGBoostGaussian(
        n_estimators=ngboost_rounds, learning_rate=ngboost_lr, score="crps", seed=int(seed)
    ).fit(Xtr, ytr)
    torf = TORFForecaster(flow=OddResidualFlow(epochs=torf_epochs, seed=int(seed))).fit(Xtr, ytr)

    diffpts_crps = diffpts.crps(Xte, yte, n_samples=n_samples, seed=int(seed) + 1)
    ngb_crps = float(ngb.crps(Xte, yte))
    torf_crps = float(torf.crps(Xte, yte))
    q = diffpts.predict_quantiles(
        Xte, np.array([0.05, 0.95]), n_samples=n_samples, seed=int(seed) + 1
    )
    samples = diffpts.predict_samples(Xte, min(n_samples, 200), seed=int(seed) + 2)
    mu = diffpts.predict(Xte)
    pit_stat, pit_p = pit_ks(diffpts.pit(Xte, yte, n_samples=n_samples, seed=int(seed) + 1))
    return {
        "synthetic_diffpts_crps": diffpts_crps,
        "synthetic_diffpts_endpoint_crps": diffpts.endpoint_crps(Xte, yte),
        "synthetic_ngboost_crps": ngb_crps,
        "synthetic_torf_crps": torf_crps,
        "synthetic_crps_gain_vs_ngboost": ngb_crps - diffpts_crps,
        "synthetic_crps_gain_vs_torf": torf_crps - diffpts_crps,
        "synthetic_diffpts_endpoint_log_score": diffpts.endpoint_log_score(Xte, yte),
        "synthetic_ngboost_log_score": float(ngb.log_score(Xte, yte)),
        "synthetic_diffpts_mae": float(np.mean(np.abs(yte - mu))),
        "synthetic_sample_mean_dev": float(np.mean(samples - mu[:, None])),
        "synthetic_coverage_90": float(np.mean((yte >= q[:, 0]) & (yte <= q[:, 1]))),
        "synthetic_pit_ks": float(pit_stat),
        "synthetic_pit_ks_pvalue": float(pit_p),
        "synthetic_n_train": float(n_train),
        "synthetic_n_test": float(n_test),
        "synthetic_seed": float(seed),
        "synthetic_noise": str(noise),
        "synthetic_schedule": str(schedule),
        "synthetic_dgp": "fixture",
        "synthetic_claim": "research_metric_only",
        "synthetic_synthetic": "heteroskedastic_symmetric_seeded",
    }
