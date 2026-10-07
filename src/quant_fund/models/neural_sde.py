"""Latent neural SDE probabilistic forecaster (prior/posterior pair + ELBO).

Latent stochastic differential equations for amortized, distribution-valued
forecasting of a univariate series on a fixed grid: an encoder compresses the
observed context window, a variational posterior SDE explains the observed
path, a prior SDE carries the generative dynamics, and a Girsanov path-space
KL keeps the posterior close to the prior. Forecasts are seeded Monte-Carlo
paths: filter the context through the posterior dynamics, roll the prior (or
posterior) drift out over the horizon, decode each latent point to a Gaussian
observation law.

References (verified against arXiv listing pages, 30 Sep 2026 — the lane spec
miscited two of the three anchors and got both author lists wrong; the REAL
papers are below and are what is implemented):

- Li, X., Wong, T.-K.L., Chen, R.T.Q. & Duvenaud, D. (2020), "Scalable
  Gradients for Stochastic Differential Equations" (the Latent-SDE paper),
  AISTATS 2020 / PMLR v118, arXiv:2001.01328 (NOT arXiv:2001.04328, which is
  Ma's "Kodaira dimension of universal holomorphic symplectic varieties", and
  the spec's author list "Li, Du & van den Hengel" is wrong). Implemented:
  Section 5's prior/posterior SDE pair

      ``dZ_t = h_0(Z_t, t) dt + sigma(Z_t, t) dW_t``          (prior)
      ``dZ_t = h_1(Z_t, t) dt + sigma(Z_t, t) dW_t``          (posterior)

  with SHARED diffusion (equivalent measures on path space — required for a
  finite Girsanov KL), the pathwise KL integrand
  ``u(z, t) = sigma(z, t)^{-1} (h_1 - h_0)(z, t)`` giving

      ``KL(q_path || p_path) = E_q[ int 1/2 ||u||^2 dt ]``,

  discretized on the Euler–Maruyama grid as ``sum_k 1/2 ||u(z_k)||^2 dt``
  (the paper's Eqs. 4-7), the reparameterized initial posterior
  ``z_0 ~ N(mu_phi(ctx), sigma_phi^2(ctx))`` vs a fixed standard-normal prior
  ``p_0`` (analytic Gaussian KL), and the ELBO

      ``ELBO = E_q[ sum_k log p(x_k | z_k, t_k) ] - KL_0 - KL_path``.

- Kidger, P., Foster, J., Li, X., Oberhauser, H. & Lyons, T. (2021), "Neural
  SDEs as Infinite-Dimensional GANs", ICML 2021 / PMLR v139:5453-5463,
  arXiv:2102.03657 (NOT arXiv:2002.09329, which is Zhang et al.'s atomic
  physics paper "Probing atomic 'quantum grating'..." — the spec's id and
  year were wrong). Not a loss contribution here — its generator side is the
  same neural SDE family; cited for the correct anchor.
- Kidger, P., Morrill, J., Foster, J. & Lyons, T. (2020), "Neural Controlled
  Differential Equations for Irregular Time Series", NeurIPS 2020,
  arXiv:2005.08926 (verified — the one spec citation that was right). The
  CDE/log-signature lineage motivates the signature encoder below.
- Krishnan, R.G., Shalit, U. & Sontag, D. (2017), "Structured Inference
  Networks for Nonlinear State Space Models", AAAI, arXiv:1609.09869 — the
  filter-then-rollout forecast construction (the latent SDE's discrete-time
  ancestor the paper cites).
- Kingma, D.P. & Welling, M. (2014), "Auto-Encoding Variational Bayes",
  ICLR, arXiv:1312.6114 — ELBO/reparameterization.
- Kloeden, P.E. & Platen, E. (1992), *Numerical Solution of Stochastic
  Differential Equations*, Springer — the Euler–Maruyama scheme (strong order
  0.5, weak order 1.0).
- Oksendal, B. (2003), *Stochastic Differential Equations*, 6th ed., Springer
  — Girsanov's theorem (the path-space KL above).
- Ornstein & Uhlenbeck (1930), Phys. Rev. 36:823; Black & Scholes (1973),
  J. Political Economy 81:637; Hamilton (1989), Econometrica 57:357 — the
  SYNTHETIC bench DGPs (OU exact transition, log-GBM exact transition,
  two-state Markov regime drift with an exact HMM-filtered predictive).
- Gneiting, T. & Raftery, A.E. (2007), JASA 102:359-378 — CRPS.

Composition (imports, not reimplementation): the lazy ``_torch`` pattern of
``models/deep_hedging.py`` (torch is the optional ``nn`` extra); context
features from ``models/path_signatures.logsignature`` (the module already
owns Lyndon-basis log-signatures — a signature encoder is the right free-
Lie-algebra summary of a context window and would be a poor reimplementation
target); proper scores from ``metrics/scoring.py`` (``crps_empirical``,
``crps_gaussian``, ``crps_gaussian_mixture``).

Deviations from the papers (documented, lane-driven):

1. FORECASTING adaptation: the paper's latent SDE encodes/reconstructs ONE
   observed span; here each training path is split at ``context_len`` — the
   encoder sees the context only, the decoder is charged for ALL grid points
   (context + horizon), so the ELBO's likelihood IS a forecasting loss and
   the path KL regularizes how far the context-conditioned posterior drift
   may depart from the prior per unit time. At inference the context is
   itself observed data, so ``predict_*`` rolls the POSTERIOR drift
   ``h_1 = f + u(z, t; ctx)`` (the amortized, data-conditional dynamics the
   ELBO trains for the horizon) from ``z_0 ~ q_0(ctx)``; ``drift="prior"``
   exposes the paper's pure-generative sampling mode for consistency checks.
2. Solver: fixed-step Euler–Maruyama with direct pathwise gradients through
   the solve (no stochastic adjoint — the paper's memory trick is a
   computational, not statistical, device; budgets here are tiny). All
   inference runs in numpy on extracted float64 parameters.
3. Encoder: logsignature(order 3) of the time-augmented standardized context
   + level scalars → MLP context vector (the paper uses a GRU); one vector
   per context, amortizing both the initial posterior and the drift
   correction ``u(z, t; ctx)``.
4. Gaussian decoder ``x_k | z_k ~ N(m_dec(z_k, t_k), s_dec^2(z_k, t_k))`` on
   a standardized univariate target; the paper's experiment section likewise
   uses Gaussian observation models.
5. Determinism: every random draw (network init excepted — seeded
   ``torch.manual_seed``) comes from a seeded numpy ``Generator`` in a fixed
   draw order; repeated calls on the same machine are bit-identical.

Honesty (AGENTS.md contract): every result here is SYNTHETIC — simulated
SDE paths with KNOWN transition laws (OU, log-GBM, regime-switch drift),
used for correctness evidence only, never market evidence. Scores are
proper (CRPS — empirical and exact mixture forms — PIT, coverage/width);
no Sharpe/Sortino/Calmar/P&L headline; no live-trading claims.

Conventions: torch is the optional ``nn`` extra, imported lazily via
:func:`_torch`, so this module imports cleanly without it; every training
entry point then raises ``ImportError`` with install guidance while the
numpy core (features, EM rollout, scoring wiring, all input-contract edges)
keeps working. Fail-closed: invalid hyperparameters, non-finite or
mis-shaped inputs, too-few paths, zero-variance targets, unfitted use,
non-finite training loss, and degenerate posteriors raise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm as _norm

from quant_fund.metrics.scoring import (
    crps_empirical,
    crps_gaussian,
    crps_gaussian_mixture,
)
from quant_fund.models.path_signatures import logsignature

Array = NDArray[np.float64]

__all__ = [
    "LatentSDEFitInfo",
    "NeuralSDEForecaster",
    "bench_neural_sde",
]

_DGPS = ("ou", "gbm", "regime")
_DRIFTS = ("posterior", "prior")
_MIN_PATHS = 8
_MIN_CONTEXT = 2
# Decoder log-scale clip (repo log-scale convention, cf. diffusion_forecaster
# / ngboost_lite); keeps the Gaussian decoder scale in [e^-c, e^c].
_LOG_S_CLIP = 12.0
# Initial diffusion level (standardized units): sigma_0 = 0.25 via the
# softplus-inverse bias, matching the "generative start" convention of the
# DiffPTS lane's zero-init endpoint.
_SIGMA_INIT = 0.25
_SIG_ORDER_MAX = 4  # mirrors path_signatures._MAX_LOGSIGNATURE_ORDER

_Layers = tuple[tuple[Array, Array], ...]


def _torch() -> Any:
    """Lazily import torch (optional ``nn`` extra); fail closed with guidance."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "the latent SDE trainer needs the optional 'nn' extra (torch): uv sync --extra nn"
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


def _check_nonnegative(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be non-negative and finite; got {value!r}")
    return v


def _check_choice(value: str, allowed: tuple[str, ...], name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}; got {value!r}")
    return value


def _check_paths(paths: Array, name: str = "paths") -> Array:
    arr = np.asarray(paths, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_paths, T); got ndim={arr.ndim}")
    if arr.shape[0] < 1 or arr.shape[1] < 2:
        raise ValueError(f"{name} must be non-empty with T >= 2; got {arr.shape}")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    return arr


def _check_contexts(contexts: Array, context_len: int) -> Array:
    arr = np.asarray(contexts, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != context_len:
        raise ValueError(f"contexts must be (n, context_len={context_len}); got shape {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError("contexts must be non-empty")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError("contexts must be finite (NaN/inf rejected)")
    return np.asarray(arr, dtype=float)


def _check_futures(Y: Array, n: int, h: int, name: str = "Y") -> Array:
    arr = np.asarray(Y, dtype=float)
    if arr.ndim != 2 or arr.shape != (n, h):
        raise ValueError(f"{name} must have shape ({n}, {h}); got {arr.shape}")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    return arr


def _require_finite_loss(value: float, epoch: int) -> float:
    """Fail-closed guard on the training loss (mirrors deep_bsde/diffusion)."""
    if not math.isfinite(value):
        raise ValueError(
            f"latent SDE training loss is not finite at epoch {epoch} (loss={value!r}); "
            "reduce lr or check inputs"
        )
    return value


def _softplus(x: Array) -> Array:
    """Numerically stable softplus (numpy mirror of torch.nn.functional)."""
    return np.asarray(np.logaddexp(0.0, x), dtype=float)


def _kl_normal(mu: Array, log_s: Array) -> Array:
    """KL(N(mu, s^2) || N(0, I)) per row, s = exp(log_s): 0.5 sum_d (mu^2 + s^2 - 1 - 2 log_s)."""
    return 0.5 * np.sum(mu * mu + np.exp(2.0 * log_s) - 1.0 - 2.0 * log_s, axis=-1)


def _gaussian_logpdf(x: Array, m: Array, log_s: Array) -> Array:
    return np.asarray(
        -0.5 * ((x - m) * np.exp(-log_s)) ** 2 - log_s - 0.5 * math.log(2.0 * math.pi),
        dtype=float,
    )


# ---------------------------------------------------------------------------
# context encoder features (logsignature composition, numpy, deterministic)
# ---------------------------------------------------------------------------


def _sig_feature_count(sig_order: int) -> int:
    """Lyndon-coordinate count of logsignature(order) on a 2-D (t, x) path.

    Witt numbers on 2 letters: l_1 = 2, l_2 = 1, l_3 = 2, l_4 = 3.
    """
    table = {1: 2, 2: 3, 3: 5, 4: 8}
    return table[sig_order]


def _context_features(contexts_std: Array, context_len: int, dt: float, sig_order: int) -> Array:
    """Encoder input features for a batch of STANDARDIZED context windows.

    Per context: ``logsignature`` of the time-augmented path
    ``(t_k, x_k)_{k=0..C-1}`` (``t_k = k dt``; the piecewise-linear signature
    of :mod:`quant_fund.models.path_signatures`) plus level scalars
    ``(x_last, mean, std)`` so the encoder sees both the path *shape* (a
    reparametrization-invariant summary — slope/curvature/regime signal) and
    the raw level/location the signature is invariant to.
    """
    t_grid = np.arange(context_len, dtype=float) * dt
    feats: list[Array] = []
    for i in range(int(contexts_std.shape[0])):
        path2d = np.column_stack([t_grid, contexts_std[i]])
        sig = logsignature(path2d, sig_order)
        x = contexts_std[i]
        scalars = np.array([x[-1], float(np.mean(x)), float(np.std(x))])
        feats.append(np.concatenate([sig, scalars]))
    return np.asarray(np.stack(feats), dtype=float)


# ---------------------------------------------------------------------------
# numpy inference core (pure numpy on extracted float64 parameters)
# ---------------------------------------------------------------------------


def _np_mlp(x: Array, layers: _Layers) -> Array:
    """Numpy mirror of the training MLPs: tanh hidden layers, linear output."""
    z = np.asarray(x, dtype=float)
    last = len(layers) - 1
    for i, (W, b) in enumerate(layers):
        z = z @ W.T + b
        if i < last:
            z = np.tanh(z)
    return np.asarray(z, dtype=float)


def _np_linear(x: Array, W: Array, b: Array) -> Array:
    return np.asarray(x @ W.T + b, dtype=float)


@dataclass(frozen=True, eq=False)
class _NumpyLatentSDEParams:
    """Float64 inference parameters extracted from the torch training run."""

    context_len: int
    horizon: int
    latent_dim: int
    sig_order: int
    dt: float  # model grid step, 1 / (context_len + horizon - 1)
    sigma_floor: float
    log_s_clip: float
    y_mean: float
    y_std: float
    feat_mean: Array
    feat_std: Array
    enc_layers: _Layers  # ctx = mlp(features)
    q0_layer: tuple[Array, Array]  # ctx -> [mu0 | log_s0]
    f_layers: _Layers  # prior drift h_0(z, t)
    u_layers: _Layers  # correction u(z, t, ctx); posterior h_1 = f + u
    sig_layers: _Layers  # shared diffusion sigma(z, t)
    dec_layers: _Layers  # z, t -> [m | log s]


def _np_encode(params: _NumpyLatentSDEParams, contexts_std: Array) -> Array:
    """Context windows -> ctx vectors (n, ctx_dim), standardized features."""
    feats = _context_features(contexts_std, params.context_len, params.dt, params.sig_order)
    feats = (feats - params.feat_mean) / params.feat_std
    return _np_mlp(feats, params.enc_layers)


def _np_q0(params: _NumpyLatentSDEParams, ctx: Array) -> tuple[Array, Array]:
    """Posterior initial law q_0(ctx) -> (mu0, log_s0); log_s0 clipped."""
    out = _np_linear(ctx, params.q0_layer[0], params.q0_layer[1])
    d = params.latent_dim
    mu0 = out[:, :d]
    log_s0 = np.clip(out[:, d:], -params.log_s_clip, params.log_s_clip)
    return np.asarray(mu0, dtype=float), np.asarray(log_s0, dtype=float)


def _zt_inputs(z: Array, t: float) -> Array:
    """Broadcast-append the time channel: (..., d_z) -> (..., d_z + 1)."""
    tb = np.broadcast_to(np.asarray(t, dtype=float), z.shape[:-1])
    return np.asarray(np.concatenate([z, tb[..., None]], axis=-1), dtype=float)


def _np_prior_drift(params: _NumpyLatentSDEParams, z: Array, t: float) -> Array:
    return _np_mlp(_zt_inputs(z, t), params.f_layers)


def _np_correction(params: _NumpyLatentSDEParams, z: Array, t: float, ctx: Array) -> Array:
    """Posterior drift correction u(z, t, ctx) (the Girsanov numerator).

    ``ctx`` (n, ctx_dim) broadcasts across the sample axis of ``z``
    (n, M, d_z) or (n, d_z).
    """
    n = ctx.shape[0]
    target = z.shape[:-1]
    ctx_b = np.broadcast_to(
        ctx.reshape((n,) + (1,) * (z.ndim - 2) + (ctx.shape[1],)),
        target + (ctx.shape[1],),
    )
    inp = np.concatenate([_zt_inputs(z, t), ctx_b], axis=-1)
    return _np_mlp(inp, params.u_layers)


def _np_diffusion(params: _NumpyLatentSDEParams, z: Array, t: float) -> Array:
    """Shared diffusion sigma(z, t) = sigma_floor + softplus(net); > 0."""
    raw = _np_mlp(_zt_inputs(z, t), params.sig_layers)
    return np.asarray(params.sigma_floor + _softplus(raw), dtype=float)


def _np_decode(params: _NumpyLatentSDEParams, z: Array, t: Array | float) -> tuple[Array, Array]:
    """Decoder law x | z, t -> (mean, log_scale) with clipped log_scale.

    ``z`` may be (n, M, d) with ``t`` (m,) broadcast over the last-but-latent
    axes: the horizon grid times vary per decoded step.
    """
    if np.ndim(t) == 0:
        tb = np.broadcast_to(np.asarray(t, dtype=float), z.shape[:-1])
    else:
        tarr = np.asarray(t, dtype=float)
        tb = np.broadcast_to(
            tarr.reshape(tarr.shape + (1,) * (z.ndim - 1 - tarr.ndim)),
            z.shape[:-1],
        )
    inp = np.concatenate([z, tb[..., None]], axis=-1)
    out = _np_mlp(inp.reshape(-1, inp.shape[-1]), params.dec_layers).reshape(z.shape[:-1] + (2,))
    m = out[..., 0]
    log_s = np.clip(out[..., 1], -params.log_s_clip, params.log_s_clip)
    return np.asarray(m, dtype=float), np.asarray(log_s, dtype=float)


def _girsanov_kl_path(
    params: _NumpyLatentSDEParams,
    z_path: Array,
    ctx: Array,
) -> Array:
    """EM-grid path-space KL between posterior and prior measures.

    On a posterior path ``z_0..z_{T-1}``:

    ``KL = sum_{k=0}^{T-2} 1/2 ||u(z_k, t_k, ctx) / sigma(z_k, t_k)||^2 dt``

    the discretized Girsanov integrand of the latent-SDE ELBO (u = h_1 - h_0;
    the shared diffusion rescales it — Oksendal 2003; Li et al. 2020 Eq. 7).
    ``z_path`` is (..., T, d_z); returns (...) per-path KL. Deterministic
    given the path — no Monte-Carlo term inside the integral.
    """
    T = int(z_path.shape[-2])
    if T < 2:
        raise ValueError(f"z_path needs >= 2 grid points; got {T}")
    kl = np.zeros(z_path.shape[:-2], dtype=float)
    for k in range(T - 1):
        t = k * params.dt
        u = _np_correction(params, z_path[..., k, :], t, ctx)
        s = _np_diffusion(params, z_path[..., k, :], t)
        kl = kl + 0.5 * np.sum((u / s) ** 2, axis=-1) * params.dt
    return np.asarray(kl, dtype=float)


def _np_rollout(
    params: _NumpyLatentSDEParams,
    contexts_std: Array,
    n_ahead: int,
    n_samples: int,
    seed: int,
    drift: str,
) -> tuple[Array, Array, Array, Array]:
    """Seeded MC path sampler: filter context under the posterior, roll the
    horizon under the chosen drift, decode each horizon grid point.

    Returns ``(z_horizon, mu_raw, sigma_raw, eta)`` with ``z_horizon``
    (n, n_ahead, M, d_z) the latent draws at grid points ``C..C+n_ahead-1``,
    ``mu_raw``/``sigma_raw`` (n, n_ahead, M) the RAW-unit decoder parameters
    — a Gaussian-mixture predictive (equal weights 1/M) at each horizon
    step, closed-form scorable before decode noise — and ``eta`` the decode
    noise (same shape) so the sampled law ``mu + sigma * eta`` shares the
    exact latent path set of the analytic mixture.

    Draw order (one seeded ``default_rng``, fixed for determinism): initial
    reparameterization noise ``eps0``, then one Brownian increment per
    context-filtering step ``k = 0..C-2``, then one per horizon step, then
    the decode noise ``eta`` LAST — a single stream, so no value is ever
    reused between the latent and decode draws.
    """
    n = int(contexts_std.shape[0])
    d = params.latent_dim
    rng = np.random.default_rng(int(seed))
    ctx = _np_encode(params, contexts_std)
    mu0, log_s0 = _np_q0(params, ctx)
    eps0 = rng.standard_normal((n, n_samples, d))
    z = mu0[:, None, :] + np.exp(log_s0)[:, None, :] * eps0
    sqrt_dt = math.sqrt(params.dt)
    # context filtering: posterior drift h_1 = f + u (context is observed data)
    for k in range(params.context_len - 1):
        t = k * params.dt
        f = _np_prior_drift(params, z, t)
        u = _np_correction(params, z, t, ctx)
        s = _np_diffusion(params, z, t)
        dW = rng.standard_normal((n, n_samples, d))
        z = z + (f + u) * params.dt + s * sqrt_dt * dW
    # horizon rollout: chosen drift (posterior = amortized forecast dynamics;
    # prior = the paper's unconditional generation mode)
    zh: list[Array] = []
    for j in range(n_ahead):
        t = (params.context_len - 1 + j) * params.dt
        f = _np_prior_drift(params, z, t)
        if drift == "posterior":
            u = _np_correction(params, z, t, ctx)
            step_drift = f + u
        else:
            step_drift = f
        s = _np_diffusion(params, z, t)
        dW = rng.standard_normal((n, n_samples, d))
        z = z + step_drift * params.dt + s * sqrt_dt * dW
        zh.append(z)
    z_h = np.stack(zh, axis=1)  # (n, n_ahead, M, d)
    t_hor = (params.context_len + np.arange(n_ahead)) * params.dt
    m_steps = []
    ls_steps = []
    for j in range(n_ahead):
        m_j, ls_j = _np_decode(params, z_h[:, j], t_hor[j])
        m_steps.append(m_j)
        ls_steps.append(ls_j)
    m_s = np.stack(m_steps, axis=1)  # (n, n_ahead, M)
    log_s = np.stack(ls_steps, axis=1)
    mu_raw = params.y_mean + params.y_std * m_s
    sig_raw = params.y_std * np.exp(log_s)
    eta = rng.standard_normal(mu_raw.shape)
    return (
        np.asarray(z_h, dtype=float),
        np.asarray(mu_raw, dtype=float),
        np.asarray(sig_raw, dtype=float),
        np.asarray(eta, dtype=float),
    )


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
    """Zero the output head of an MLP (or a bare Linear) — generative start."""
    last = net[-1] if isinstance(net, torch.nn.Sequential) else net
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


def _train_latent_sde(
    torch: Any,
    feats: Array,
    ys: Array,
    *,
    context_len: int,
    latent_dim: int,
    ctx_dim: int,
    hidden: Sequence[int],
    dt: float,
    sigma_floor: float,
    log_s_clip: float,
    kl_weight: float,
    epochs: int,
    lr: float,
    batch_size: int | None,
    seed: int,
) -> tuple[Any, Any, Any, Any, Any, Any, list[float], list[float], list[float], list[float]]:
    """Adam on the negated latent-SDE ELBO; deterministic given ``seed``.

    Per update (all noise from a seeded numpy Generator in fixed order —
    batch permutation, then ``eps0``, then the per-step Brownian increments —
    so init and training noise never interleave and runs are bit-identical):

    ``z_0 = mu_0(ctx) + sigma_0(ctx) * eps0`` (reparameterized posterior
    init), EM posterior solve ``z_{k+1} = z_k + (f+u) dt + sigma sqrt(dt)
    eps_k`` over the FULL grid, Gaussian decoder log-likelihood on all T
    observed points, path KL ``sum_k 1/2 ||u/sigma||^2 dt``, and analytic
    ``KL(q_0 || N(0, I))``. ``loss = NLL + kl_weight * (KL_0 + KL_path)``.

    Init: all output heads zeroed except the diffusion net's bias, which is
    set to softplus^{-1}(sigma_init - floor) — so at epoch 0 f == 0, u == 0,
    q_0 == N(0, I) (every KL is exactly 0) and the decoder is N(0, 1); the
    recorded entry-0 loss equals the closed form T/2 (mean y_s^2 + log 2pi).
    """
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    n, T = int(feats.shape[0]), int(ys.shape[1])
    d = int(latent_dim)
    enc_net = _build_mlp(torch, int(feats.shape[1]), (32,), ctx_dim)
    q0_net = torch.nn.Linear(ctx_dim, 2 * d)
    f_net = _build_mlp(torch, d + 1, hidden, d)
    u_net = _build_mlp(torch, d + 1 + ctx_dim, hidden, d)
    sig_net = _build_mlp(torch, d + 1, hidden, d)
    dec_net = _build_mlp(torch, d + 1, hidden, 2)
    for net in (q0_net, f_net, u_net, dec_net):
        _init_zero_last(torch, net)
    with torch.no_grad():
        b = sig_net[-1].bias
        b.fill_(math.log(math.expm1(_SIGMA_INIT - sigma_floor)))
    params: list[Any] = []
    for net in (enc_net, q0_net, f_net, u_net, sig_net, dec_net):
        params.extend(net.parameters())
    opt = torch.optim.Adam(params, lr=float(lr))
    bsz = n if batch_size is None else min(int(batch_size), n)
    steps_per_epoch = max(1, -(-n // bsz))
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(epochs) * steps_per_epoch, eta_min=lr / 10.0
    )
    feats_all = torch.as_tensor(feats, dtype=torch.float32)
    y_all = torch.as_tensor(ys, dtype=torch.float32)
    t_grid = torch.as_tensor(np.arange(T, dtype=float) * dt, dtype=torch.float32)
    sqrt_dt = math.sqrt(float(dt))
    log_2pi_half = 0.5 * math.log(2.0 * math.pi)
    rng = np.random.default_rng(int(seed))
    loss_curve: list[float] = []
    nll_curve: list[float] = []
    kl0_curve: list[float] = []
    klp_curve: list[float] = []
    update = 0
    for _epoch in range(int(epochs)):
        order = np.arange(n) if batch_size is None else rng.permutation(n)
        for start in range(0, n, bsz):
            idx = order[start : start + bsz]
            nb = int(idx.size)
            f_b = feats_all if batch_size is None else feats_all[idx]
            y_b = y_all if batch_size is None else y_all[idx]
            eps0 = torch.as_tensor(rng.standard_normal((nb, d)), dtype=torch.float32)
            dW = torch.as_tensor(rng.standard_normal((nb, T - 1, d)), dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            ctx = enc_net(f_b)
            q0 = q0_net(ctx)
            mu0 = q0[:, :d]
            log_s0 = torch.clamp(q0[:, d:], -float(log_s_clip), float(log_s_clip))
            kl0 = 0.5 * torch.sum(mu0 * mu0 + torch.exp(2.0 * log_s0) - 1.0 - 2.0 * log_s0, dim=-1)
            z = mu0 + torch.exp(log_s0) * eps0
            z_path: list[Any] = [z]
            kl_path = torch.zeros(nb, dtype=torch.float32)
            for k in range(T - 1):
                t_fill = torch.full((nb, 1), float(k) * dt, dtype=torch.float32)
                zt = torch.cat([z, t_fill], dim=-1)
                f = f_net(zt)
                u = u_net(torch.cat([zt, ctx], dim=-1))
                s = sigma_floor + torch.nn.functional.softplus(sig_net(zt))
                kl_path = kl_path + 0.5 * torch.sum((u / s) ** 2, dim=-1) * dt
                z = z + (f + u) * dt + s * sqrt_dt * dW[:, k, :]
                z_path.append(z)
            z_all = torch.stack(z_path, dim=1)  # (nb, T, d)
            t_all = t_grid.view(1, T, 1).expand(nb, T, 1)
            dec = dec_net(torch.cat([z_all, t_all], dim=-1))  # (nb, T, 2)
            m = dec[..., 0]
            log_s = torch.clamp(dec[..., 1], -float(log_s_clip), float(log_s_clip))
            resid = (y_b - m) * torch.exp(-log_s)
            nll = torch.sum(0.5 * resid * resid + log_s + log_2pi_half, dim=-1)
            kl_tot = kl0 + kl_path
            loss = torch.mean(nll + float(kl_weight) * kl_tot)
            loss_curve.append(_require_finite_loss(float(loss.detach().numpy()), update))
            nll_curve.append(float(torch.mean(nll).detach().numpy()))
            kl0_curve.append(float(torch.mean(kl0).detach().numpy()))
            klp_curve.append(float(torch.mean(kl_path).detach().numpy()))
            loss.backward()
            opt.step()
            lr_sched.step()
            update += 1
    return (
        enc_net,
        q0_net,
        f_net,
        u_net,
        sig_net,
        dec_net,
        loss_curve,
        nll_curve,
        kl0_curve,
        klp_curve,
    )


# ---------------------------------------------------------------------------
# the forecaster
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatentSDEFitInfo:
    """Training trace (curves recorded BEFORE each Adam update).

    Entry 0 is the exact-ELBO loss at the engineered init: f == 0, u == 0,
    ``q_0 == p_0 == N(0, I)`` (both KL terms vanish) and decoder N(0, 1), so
    ``loss_curve[0] == T/2 * (mean(y_s^2) + log 2*pi)`` up to float32 casts —
    the tests assert that closed form. ``final_elbo`` is ``-loss_curve[-1]``.
    """

    context_len: int
    horizon: int
    latent_dim: int
    sig_order: int
    epochs: int
    kl_weight: float
    seed: int
    n_paths: int
    loss_curve: list[float]
    nll_curve: list[float]
    kl0_curve: list[float]
    kl_path_curve: list[float]
    final_elbo: float
    final_kl_path: float


class NeuralSDEForecaster:
    """Latent-SDE probabilistic forecaster (Li et al. 2020, arXiv:2001.01328).

    ``fit(paths)`` trains the prior/posterior pair on a batch of SYNTHETIC or
    observed paths of length ``context_len + horizon`` by maximizing the ELBO;
    ``predict_*`` draws seeded Monte-Carlo continuations: posterior-drift EM
    filtering through the context grid, then rollout over the horizon and
    Gaussian decode per step (an (n, H, M) equal-weight Gaussian mixture
    before the decode noise — :meth:`predict_components` exposes it for
    zero-decode-noise exact scoring). ``fit`` needs torch (the ``nn`` extra);
    every inference method is pure numpy on extracted float64 parameters.
    """

    def __init__(
        self,
        context_len: int = 8,
        horizon: int = 4,
        latent_dim: int = 2,
        sig_order: int = 3,
        ctx_dim: int = 24,
        hidden: Sequence[int] = (48, 48),
        sigma_floor: float = 1e-3,
        log_s_clip: float = _LOG_S_CLIP,
        kl_weight: float = 1.0,
        epochs: int = 200,
        lr: float = 3e-3,
        batch_size: int | None = 128,
        seed: int = 0,
    ) -> None:
        if isinstance(context_len, bool) or int(context_len) != context_len:
            raise ValueError(f"context_len must be an int; got {context_len!r}")
        if int(context_len) < _MIN_CONTEXT:
            raise ValueError(
                f"context_len must be >= {_MIN_CONTEXT} (the encoder needs a path); "
                f"got {context_len!r}"
            )
        if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) < 1:
            raise ValueError(f"horizon must be an int >= 1; got {horizon!r}")
        if isinstance(latent_dim, bool) or int(latent_dim) != latent_dim or int(latent_dim) < 1:
            raise ValueError(f"latent_dim must be an int >= 1; got {latent_dim!r}")
        if (
            isinstance(sig_order, bool)
            or int(sig_order) != sig_order
            or not 1 <= int(sig_order) <= _SIG_ORDER_MAX
        ):
            raise ValueError(
                f"sig_order must be an int in [1, {_SIG_ORDER_MAX}]; got {sig_order!r}"
            )
        if isinstance(ctx_dim, bool) or int(ctx_dim) != ctx_dim or int(ctx_dim) < 1:
            raise ValueError(f"ctx_dim must be an int >= 1; got {ctx_dim!r}")
        hidden_widths = tuple(int(hh) for hh in hidden)
        if not hidden_widths or any(hh < 1 for hh in hidden_widths):
            raise ValueError(
                f"hidden must be a non-empty sequence of positive widths; got {hidden!r}"
            )
        if isinstance(epochs, bool) or int(epochs) != epochs or int(epochs) < 1:
            raise ValueError(f"epochs must be an int >= 1; got {epochs!r}")
        if batch_size is not None and (
            isinstance(batch_size, bool) or int(batch_size) != batch_size or int(batch_size) < 1
        ):
            raise ValueError(f"batch_size must be an int >= 1 or None; got {batch_size!r}")
        self.context_len = int(context_len)
        self.horizon = int(horizon)
        self.latent_dim = int(latent_dim)
        self.sig_order = int(sig_order)
        self.ctx_dim = int(ctx_dim)
        self.hidden = hidden_widths
        self.sigma_floor = _check_positive(sigma_floor, "sigma_floor")
        if self.sigma_floor >= _SIGMA_INIT:
            raise ValueError(
                f"sigma_floor must be < {_SIGMA_INIT} (the init level); got {sigma_floor!r}"
            )
        self.log_s_clip = _check_positive(log_s_clip, "log_s_clip")
        self.kl_weight = _check_nonnegative(kl_weight, "kl_weight")
        self.epochs = int(epochs)
        self.lr = _check_positive(lr, "lr")
        self.batch_size = None if batch_size is None else int(batch_size)
        self.seed = int(seed)
        self._params: _NumpyLatentSDEParams | None = None
        self.fit_info: LatentSDEFitInfo | None = None

    # -- state ---------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._params is not None

    def _require_params(self) -> _NumpyLatentSDEParams:
        if self._params is None:
            raise RuntimeError("NeuralSDEForecaster is not fitted")
        return self._params

    def _standardize_contexts(self, contexts: Array) -> Array:
        params = self._require_params()
        arr = _check_contexts(contexts, params.context_len)
        return np.asarray((arr - params.y_mean) / params.y_std, dtype=float)

    # -- training (torch-gated) ------------------------------------------------

    def fit(self, paths: Array) -> NeuralSDEForecaster:
        """Fit the latent-SDE ELBO on a batch of observed paths.

        ``paths`` is (n_paths, T) with ``T == context_len + horizon``: the
        first ``context_len`` points form the encoded context, the decoder is
        charged on all T. Inputs are validated BEFORE torch is touched, so
        input-contract errors raise ``ValueError`` even without the ``nn``
        extra; a successful fit needs torch and raises ``ImportError`` with
        guidance when absent.
        """
        P = _check_paths(paths)
        T = self.context_len + self.horizon
        if P.shape[1] != T:
            raise ValueError(f"paths must have T == context_len + horizon == {T}; got {P.shape[1]}")
        if P.shape[0] < _MIN_PATHS:
            raise ValueError(f"too few paths: need >= {_MIN_PATHS}, got {P.shape[0]}")
        y_std = float(np.std(P))
        if not math.isfinite(y_std) or y_std <= 0.0:
            raise ValueError("paths have zero variance (constant series); nothing to fit")
        y_mean = float(np.mean(P))
        Ps = np.asarray((P - y_mean) / y_std, dtype=float)
        dt = 1.0 / float(T - 1)
        contexts_std = Ps[:, : self.context_len]
        feats = _context_features(contexts_std, self.context_len, dt, self.sig_order)
        feat_mean = feats.mean(axis=0)
        feat_std = feats.std(axis=0)
        feat_std = np.where(feat_std > 0.0, feat_std, 1.0)
        feats_std = np.asarray((feats - feat_mean) / feat_std, dtype=float)

        torch = _torch()
        (
            enc_net,
            q0_net,
            f_net,
            u_net,
            sig_net,
            dec_net,
            loss_curve,
            nll_curve,
            kl0_curve,
            klp_curve,
        ) = _train_latent_sde(
            torch,
            feats_std,
            Ps,
            context_len=self.context_len,
            latent_dim=self.latent_dim,
            ctx_dim=self.ctx_dim,
            hidden=self.hidden,
            dt=dt,
            sigma_floor=self.sigma_floor,
            log_s_clip=self.log_s_clip,
            kl_weight=self.kl_weight,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            seed=self.seed,
        )
        q0_W = np.array(q0_net.weight.detach().numpy(), dtype=float)
        q0_b = np.array(q0_net.bias.detach().numpy(), dtype=float)
        params = _NumpyLatentSDEParams(
            context_len=self.context_len,
            horizon=self.horizon,
            latent_dim=self.latent_dim,
            sig_order=self.sig_order,
            dt=dt,
            sigma_floor=self.sigma_floor,
            log_s_clip=self.log_s_clip,
            y_mean=y_mean,
            y_std=y_std,
            feat_mean=np.asarray(feat_mean, dtype=float),
            feat_std=np.asarray(feat_std, dtype=float),
            enc_layers=_extract_mlp_layers(torch, enc_net),
            q0_layer=(q0_W, q0_b),
            f_layers=_extract_mlp_layers(torch, f_net),
            u_layers=_extract_mlp_layers(torch, u_net),
            sig_layers=_extract_mlp_layers(torch, sig_net),
            dec_layers=_extract_mlp_layers(torch, dec_net),
        )
        self._params = params
        _, mu_raw, sig_raw, _eta = _np_rollout(
            params,
            contexts_std[:_MIN_PATHS],
            n_ahead=1,
            n_samples=4,
            seed=int(self.seed) + 10_000,
            drift="posterior",
        )
        if not (
            bool(np.all(np.isfinite(mu_raw)))
            and bool(np.all(np.isfinite(sig_raw)))
            and bool(np.all(sig_raw > 0.0))
        ):
            self._params = None
            raise ValueError("extracted parameters produce a non-finite predictive")
        self.fit_info = LatentSDEFitInfo(
            context_len=self.context_len,
            horizon=self.horizon,
            latent_dim=self.latent_dim,
            sig_order=self.sig_order,
            epochs=self.epochs,
            kl_weight=self.kl_weight,
            seed=self.seed,
            n_paths=int(P.shape[0]),
            loss_curve=loss_curve,
            nll_curve=nll_curve,
            kl0_curve=kl0_curve,
            kl_path_curve=klp_curve,
            final_elbo=float(-loss_curve[-1]),
            final_kl_path=float(klp_curve[-1]),
        )
        return self

    # -- numpy inference -------------------------------------------------------

    def predict_components(
        self,
        contexts: Array,
        n_ahead: int | None = None,
        n_samples: int = 128,
        seed: int = 0,
        drift: str = "posterior",
    ) -> tuple[Array, Array]:
        """Analytic Gaussian-mixture predictive, RAW units.

        Returns ``(mu, sigma)`` each ``(n, n_ahead, M)``: the decoder means
        and scales of ``M`` latent path draws — at each horizon step an
        equal-weight M-component Gaussian mixture, closed-form scorable
        (``crps_gaussian_mixture``) with no decode noise.
        """
        params = self._require_params()
        h = self.horizon if n_ahead is None else _check_count(n_ahead, "n_ahead")
        m = _check_count(n_samples, "n_samples")
        _check_choice(drift, _DRIFTS, "drift")
        contexts_std = self._standardize_contexts(contexts)
        _, mu_raw, sig_raw, _eta = _np_rollout(params, contexts_std, h, m, int(seed), drift)
        if not (bool(np.all(np.isfinite(mu_raw))) and bool(np.all(sig_raw > 0.0))):
            raise ValueError("predictive components are not finite/positive")
        return mu_raw, sig_raw

    def predict_samples(
        self,
        contexts: Array,
        n_ahead: int | None = None,
        n_samples: int = 128,
        seed: int = 0,
        drift: str = "posterior",
    ) -> Array:
        """Seeded decode-noised draws ``x_{C+j}``, shape ``(n, n_ahead, M)``.

        The latent paths are the same ones :meth:`predict_components` scores
        analytically; the returned draws add one fresh Gaussian decode draw
        per (n, step, sample) from the same seeded Generator (draw order:
        initial noise, per-step Brownian increments, decode noise).
        """
        params = self._require_params()
        h = self.horizon if n_ahead is None else _check_count(n_ahead, "n_ahead")
        m = _check_count(n_samples, "n_samples")
        _check_choice(drift, _DRIFTS, "drift")
        contexts_std = self._standardize_contexts(contexts)
        _zh, mu_raw, sig_raw, eta = _np_rollout(params, contexts_std, h, m, int(seed), drift)
        samples = mu_raw + sig_raw * eta
        if not bool(np.all(np.isfinite(samples))):
            raise ValueError("predictive samples are not finite")
        return np.asarray(samples, dtype=float)

    def predict_quantiles(
        self,
        contexts: Array,
        taus: Array,
        n_ahead: int | None = None,
        n_samples: int = 256,
        seed: int = 0,
    ) -> Array:
        """Predictive quantiles from seeded samples, shape (n, n_ahead, T).

        Sample quantiles (linear interpolation) are monotone in ``tau`` for
        sorted ``taus``, so horizons never cross.
        """
        t = np.asarray(taus, dtype=float).ravel()
        if (
            t.size == 0
            or not bool(np.all(np.isfinite(t)))
            or bool(np.any(t <= 0.0))
            or bool(np.any(t >= 1.0))
        ):
            raise ValueError(f"taus must be non-empty, finite, and in (0, 1); got {taus!r}")
        samples = self.predict_samples(contexts, n_ahead, n_samples, seed)
        return np.asarray(np.quantile(samples, t, axis=2).transpose(1, 2, 0), dtype=float)

    def pit(
        self,
        contexts: Array,
        Y: Array,
        n_samples: int = 256,
        seed: int = 0,
    ) -> Array:
        """Randomized PIT per (context, horizon step), shape (n, n_ahead).

        ``F_hat(y) = (#{s < y} + 0.5 #{s = y} + 0.5) / (M + 1)`` on the
        seeded sample law, strictly inside (0, 1); uniform under a calibrated
        predictive up to the documented O(1/M) discreteness floor.
        """
        samples = self.predict_samples(contexts, None, n_samples, seed)
        yv = _check_futures(Y, samples.shape[0], samples.shape[1])
        less = np.sum(samples < yv[:, :, None], axis=2)
        equal = np.sum(samples == yv[:, :, None], axis=2)
        return np.asarray((less + 0.5 * equal + 0.5) / (n_samples + 1.0), dtype=float)

    def crps(
        self,
        contexts: Array,
        Y: Array,
        n_samples: int = 256,
        seed: int = 0,
    ) -> float:
        """Mean CRPS over (context, horizon step) from seeded samples.

        ``quant_fund.metrics.scoring.crps_empirical`` per step
        (Gneiting & Raftery 2007 ensemble form; O(1/sqrt(M)) MC noise,
        O(1/M) V-statistic bias — documented MC tolerance). For the
        zero-decode-noise score of the analytic mixture see
        :meth:`mixture_crps`.
        """
        samples = self.predict_samples(contexts, None, n_samples, seed)
        yv = _check_futures(Y, samples.shape[0], samples.shape[1])
        n, h, _m = samples.shape
        scores = [
            crps_empirical(float(yv[i, j]), samples[i, j]) for i in range(n) for j in range(h)
        ]
        out = float(np.mean(np.asarray(scores, dtype=float)))
        if not math.isfinite(out):
            raise ValueError("sample CRPS is not finite")
        return out

    def mixture_crps(
        self,
        contexts: Array,
        Y: Array,
        n_samples: int = 256,
        seed: int = 0,
    ) -> float:
        """Closed-form mean CRPS of the latent-path Gaussian mixture.

        Per (context, step): ``F = (1/M) sum_m N(mu_m, sigma_m^2)`` scored
        exactly by ``crps_gaussian_mixture`` — same latent draws as
        :meth:`predict_samples`, no decode noise (MC error is only the
        O(1/sqrt(M)) latent-path sampling error).
        """
        mu, sig = self.predict_components(contexts, None, n_samples, seed)
        n, h, m = mu.shape
        yv = _check_futures(Y, n, h)
        w = np.full(m, 1.0 / m)
        scores = [
            float(crps_gaussian_mixture(yv[i, j : j + 1], w, mu[i, j], sig[i, j])[0])
            for i in range(n)
            for j in range(h)
        ]
        out = float(np.mean(np.asarray(scores, dtype=float)))
        if not math.isfinite(out):
            raise ValueError("mixture CRPS is not finite")
        return out


# ---------------------------------------------------------------------------
# SYNTHETIC bench: known-law SDEs with exact oracle predictives
# ---------------------------------------------------------------------------


def _sim_ou_paths(
    n: int, T: int, dt: float, theta: float, mu: float, sigma: float, seed: int
) -> Array:
    """Exact-transition OU simulation (stationary init): correctness fixture.

    ``x_{t+1} = mu + (x_t - mu) e^{-theta dt} + s_dev eps`` with
    ``s_dev = sigma sqrt((1 - e^{-2 theta dt}) / (2 theta))`` and
    ``x_0 ~ N(mu, sigma^2 / (2 theta))`` — the EXACT law (no EM bias), so the
    bench's true predictive is the Gaussian transition kernel, not an
    approximation of it (Kloeden & Platen 1992 §4.4 exact OU scheme).
    """
    rng = np.random.default_rng(int(seed))
    a = math.exp(-theta * dt)
    s_dev = sigma * math.sqrt((1.0 - a * a) / (2.0 * theta))
    stat_sd = sigma / math.sqrt(2.0 * theta)
    x = np.empty((int(n), int(T)), dtype=float)
    x[:, 0] = mu + stat_sd * rng.standard_normal(n)
    for k in range(T - 1):
        x[:, k + 1] = mu + (x[:, k] - mu) * a + s_dev * rng.standard_normal(n)
    return x


def _sim_abm_paths(n: int, T: int, dt: float, mu: float, sigma: float, seed: int) -> Array:
    """Arithmetic Brownian motion (the log-price of a GBM): exact increments.

    ``x_{t+1} = x_t + mu dt + sigma sqrt(dt) eps`` — a Gaussian random walk,
    the exact transition law of ``log S`` under Black & Scholes (1973) GBM.
    """
    rng = np.random.default_rng(int(seed))
    x = np.empty((int(n), int(T)), dtype=float)
    x[:, 0] = rng.standard_normal(n) * sigma * math.sqrt(dt)
    incr = mu * dt + sigma * math.sqrt(dt) * rng.standard_normal((n, T - 1))
    x[:, 1:] = x[:, :1] + np.cumsum(incr, axis=1)
    return x


def _sim_regime_paths(
    n: int,
    T: int,
    dt: float,
    mu_states: tuple[float, float],
    sigma: float,
    stay: float,
    seed: int,
) -> tuple[Array, Array]:
    """Two-state Markov-drift random walk (Hamilton 1989), exact simulation.

    Increment ``x_{k+1} - x_k | s_k ~ N(mu_{s_k} dt, sigma^2 dt)`` with a
    sticky symmetric chain ``P(stay) = stay``. Returns ``(x, s)`` — the
    SYNTHETIC regime labels are test evidence only.
    """
    rng = np.random.default_rng(int(seed))
    s = np.empty((int(n), int(T)), dtype=int)
    s[:, 0] = rng.integers(0, 2, size=n)
    for k in range(T - 1):
        flip = rng.random(n) < (1.0 - stay)
        s[:, k + 1] = np.where(flip, 1 - s[:, k], s[:, k])
    mu_arr = np.asarray(mu_states, dtype=float)
    incr = mu_arr[s[:, : T - 1]] * dt + sigma * math.sqrt(dt) * rng.standard_normal((n, T - 1))
    x = np.empty((int(n), int(T)), dtype=float)
    x[:, 0] = sigma * math.sqrt(dt) * rng.standard_normal(n)
    x[:, 1:] = x[:, :1] + np.cumsum(incr, axis=1)
    return x, s


def _oracle_ou(
    x_last: Array, n_ahead: int, dt: float, theta: float, mu: float, sigma: float
) -> tuple[Array, Array]:
    """Exact OU predictive: x_{+k} | x_last ~ N(mu + d a^k, v_k)."""
    k = np.arange(1, n_ahead + 1, dtype=float)
    a_k = np.exp(-theta * dt * k)
    mean = mu + (x_last[:, None] - mu) * a_k[None, :]
    var = (sigma * sigma / (2.0 * theta)) * (1.0 - a_k[None, :] ** 2)
    var = np.broadcast_to(var, mean.shape)
    return np.asarray(mean, dtype=float), np.asarray(np.sqrt(var), dtype=float)


def _oracle_abm(
    x_last: Array, n_ahead: int, dt: float, mu: float, sigma: float
) -> tuple[Array, Array]:
    """Exact ABM predictive: x_{+k} | x_last ~ N(x_last + k mu dt, k sigma^2 dt)."""
    k = np.arange(1, n_ahead + 1, dtype=float)
    mean = x_last[:, None] + mu * dt * k[None, :]
    sd = sigma * math.sqrt(dt) * np.sqrt(k)[None, :] * np.ones((x_last.size, 1))
    return np.asarray(mean, dtype=float), np.asarray(sd, dtype=float)


def _oracle_regime(
    contexts: Array,
    n_ahead: int,
    dt: float,
    mu_states: tuple[float, float],
    sigma: float,
    stay: float,
) -> tuple[Array, Array, Array]:
    """EXACT regime-switch predictive via HMM filtering + path enumeration.

    The context's increments drive a discrete forward filter over the
    two-state chain (emission ``N(mu_s dt, sigma^2 dt)``); the predictive at
    horizon step k is the exact Gaussian mixture over all ``2^k`` regime
    continuations weighted by the Markov extension of the filtered state —
    returned as ``(weights, mean, sd)`` each broadcastable to per-step
    component lists: ``weights`` (n, k_max=2^n_ahead rows padded) is ragged in
    spirit, so the return is a list-shaped (n, n_ahead, K_max) array with
    zero-weight padding, and ``mean``/``sd`` match it.
    """
    mu_arr = np.asarray(mu_states, dtype=float)
    Pi = np.array([[stay, 1.0 - stay], [1.0 - stay, stay]])
    incr_sd = sigma * math.sqrt(dt)
    n = int(contexts.shape[0])
    K = 2 ** int(n_ahead)
    w_out = np.zeros((n, n_ahead, K), dtype=float)
    m_out = np.zeros((n, n_ahead, K), dtype=float)
    s_out = np.zeros((n, n_ahead, K), dtype=float)
    for i in range(n):
        # forward filter over s_k for k = 0..C-2 (regime driving increment k)
        resid = np.diff(contexts[i])
        f = np.full(2, 0.5)
        for r in resid:
            emit = np.exp(-0.5 * ((r - mu_arr * dt) / incr_sd) ** 2) / (
                incr_sd * math.sqrt(2.0 * math.pi)
            )
            f = f * emit
            tot = float(f.sum())
            if tot <= 0.0:
                f = np.full(2, 0.5)
            else:
                f = f / tot
        # state for the FIRST forecast increment is s_{C-1} ~ f_next @ ...
        # careful: filtered f covers s_{C-2}; propagate once for s_{C-1}
        p_state = f @ Pi
        for k in range(1, n_ahead + 1):
            # enumerate all regime sequences of length k by integer encoding
            w_seq = np.empty(2**k, dtype=float)
            m_seq = np.empty(2**k, dtype=float)
            for code in range(2**k):
                seq = [(code >> b) & 1 for b in range(k)]
                w = float(p_state[seq[0]])
                acc = mu_arr[seq[0]]
                prev = seq[0]
                for s_j in seq[1:]:
                    w *= float(Pi[prev, s_j])
                    acc += mu_arr[s_j]
                    prev = s_j
                w_seq[code] = w
                m_seq[code] = acc * dt
            total = float(w_seq.sum())
            w_seq = w_seq / total if total > 0 else np.full(2**k, 2.0**-k)
            w_out[i, k - 1, : 2**k] = w_seq
            m_out[i, k - 1, : 2**k] = contexts[i, -1] + m_seq
            s_out[i, k - 1, : 2**k] = incr_sd * math.sqrt(k)
    return w_out, m_out, s_out


def _gaussian_baseline(train_paths: Array, context_len: int, horizon: int) -> tuple[Array, Array]:
    """Ignorant baseline: per-step N(mean, sd) of the continuation INCREMENT
    distribution over the training batch, anchored at each eval context's
    last point — the 'no-path-information' reference the forecaster must
    beat on proper score."""
    inc = train_paths[:, context_len:] - train_paths[:, context_len - 1 : context_len]
    mu = inc.mean(axis=0)[:horizon]
    sd = inc.std(axis=0)[:horizon]
    sd = np.where(sd > 0.0, sd, 1.0)
    return np.asarray(mu, dtype=float), np.asarray(sd, dtype=float)


def _mixture_quantiles(
    weights: Array, mu: Array, sigma: Array, taus: tuple[float, ...], grid_n: int = 800
) -> Array:
    """Quantiles of a 1-D Gaussian mixture by grid-CDF inversion.

    Brackets the support at ±4 component sds around the component means,
    evaluates ``F(x) = sum_k w_k Phi((x - mu_k)/sig_k)`` on ``grid_n`` points
    and inverts by linear interpolation — deterministic, dependency-free,
    accurate to ~grid resolution (bench-only use; ``crps_gaussian_mixture``
    owns the scoring primitive and is NOT duplicated here).
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    lo = float(np.min(mu - 4.0 * sigma))
    hi = float(np.max(mu + 4.0 * sigma))
    grid = np.linspace(lo, hi, int(grid_n))
    cdf = (_norm.cdf((grid[:, None] - mu[None, :]) / sigma[None, :]) * w[None, :]).sum(axis=1)
    return np.asarray(np.interp(np.asarray(taus, dtype=float), cdf, grid), dtype=float)


def bench_neural_sde(
    dgp: str = "ou",
    n_train: int = 192,
    n_eval: int = 64,
    context_len: int = 8,
    horizon: int = 4,
    latent_dim: int = 2,
    sig_order: int = 3,
    ctx_dim: int = 24,
    hidden: Sequence[int] = (48, 48),
    epochs: int = 200,
    lr: float = 3e-3,
    batch_size: int | None = 128,
    kl_weight: float = 1.0,
    n_samples: int = 256,
    seed: int = 0,
) -> dict[str, float | str]:
    """NeuralSDEForecaster vs the EXACT oracle law on SYNTHETIC SDE paths.

    Correctness bench: fit on ``n_train`` simulated paths, forecast the
    continuation of ``n_eval`` held-out paths, and score against (a) the
    true transition kernel (OU / ABM closed-form, or the exact HMM-filtered
    regime mixture) and (b) the anchored ignorant Gaussian baseline. All
    scores are proper (CRPS — empirical-sample and closed-form mixture —
    plus PIT-KS and interval coverage/width); labeled ``dgp`` / ``synthetic``
    / ``claim=research_metric_only``; never market evidence, no Sharpe/P&L
    (AGENTS.md honesty contract). Requires the torch ``nn`` extra for the
    fit. MC slack is documented (seeded, O(1/sqrt(M))).
    """
    _check_choice(dgp, _DGPS, "dgp")
    T = int(context_len) + int(horizon)
    dt = 1.0 / float(T - 1)
    n_tot = _check_count(n_train, "n_train") + _check_count(n_eval, "n_eval")

    torch = _torch()  # fail fast before simulating when the nn extra is absent
    del torch

    if dgp == "ou":
        theta, mu_ou, sig_ou = 3.0, 0.0, 0.6
        paths = _sim_ou_paths(n_tot, T, dt, theta, mu_ou, sig_ou, seed)
        contexts = paths[n_train:, :context_len]
        y_true = paths[n_train:, context_len:]
        om, osd = _oracle_ou(contexts[:, -1], horizon, dt, theta, mu_ou, sig_ou)
        oracle_crps = float(np.mean(crps_gaussian(y_true.ravel(), om.ravel(), osd.ravel())))
    elif dgp == "gbm":
        mu_g, sig_g = 0.4, 0.5
        paths = _sim_abm_paths(n_tot, T, dt, mu_g, sig_g, seed)
        contexts = paths[n_train:, :context_len]
        y_true = paths[n_train:, context_len:]
        om, osd = _oracle_abm(contexts[:, -1], horizon, dt, mu_g, sig_g)
        oracle_crps = float(np.mean(crps_gaussian(y_true.ravel(), om.ravel(), osd.ravel())))
    else:
        mu_states, sig_r, stay = (-1.8, 1.8), 0.45, 0.9
        paths, _s = _sim_regime_paths(n_tot, T, dt, mu_states, sig_r, stay, seed)
        contexts = paths[n_train:, :context_len]
        y_true = paths[n_train:, context_len:]
        w_or, m_or, s_or = _oracle_regime(contexts, horizon, dt, mu_states, sig_r, stay)
        oracle_scores = []
        for i in range(int(contexts.shape[0])):
            for j in range(horizon):
                w = w_or[i, j]
                keep = w > 0
                oracle_scores.append(
                    float(
                        crps_gaussian_mixture(
                            y_true[i, j : j + 1],
                            w[keep] / w[keep].sum(),
                            m_or[i, j][keep],
                            s_or[i, j][keep],
                        )[0]
                    )
                )
        oracle_crps = float(np.mean(np.asarray(oracle_scores, dtype=float)))

    train = paths[:n_train]
    model = NeuralSDEForecaster(
        context_len=context_len,
        horizon=horizon,
        latent_dim=latent_dim,
        sig_order=sig_order,
        ctx_dim=ctx_dim,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        kl_weight=kl_weight,
        seed=int(seed),
    ).fit(train)

    model_crps = model.crps(contexts, y_true, n_samples=n_samples, seed=int(seed) + 1)
    mix_crps = model.mixture_crps(contexts, y_true, n_samples=n_samples, seed=int(seed) + 1)
    taus = np.array([0.05, 0.25, 0.75, 0.95])
    q = model.predict_quantiles(contexts, taus, n_samples=n_samples, seed=int(seed) + 1)
    cov50 = float(np.mean((y_true >= q[:, :, 1]) & (y_true <= q[:, :, 2])))
    cov90 = float(np.mean((y_true >= q[:, :, 0]) & (y_true <= q[:, :, 3])))
    width90 = float(np.mean(q[:, :, 3] - q[:, :, 0]))
    from quant_fund.metrics.probability import pit_ks

    pit_stat, pit_p = pit_ks(
        model.pit(contexts, y_true, n_samples=n_samples, seed=int(seed) + 1).ravel()
    )
    bm, bs = _gaussian_baseline(train, context_len, horizon)
    base_mu = contexts[:, -1:] + bm[None, :]
    base_sd = np.broadcast_to(bs[None, :], y_true.shape)
    baseline_crps = float(np.mean(crps_gaussian(y_true.ravel(), base_mu.ravel(), base_sd.ravel())))
    # oracle 90% coverage/width for scale comparison
    if dgp == "regime":
        ocov, ow = [], []
        for i in range(int(contexts.shape[0])):
            for j in range(horizon):
                w = w_or[i, j]
                keep = w > 0
                q05, q95 = _mixture_quantiles(
                    w[keep], m_or[i, j][keep], s_or[i, j][keep], (0.05, 0.95)
                )
                ow.append(q95 - q05)
                ocov.append(float(q05 <= y_true[i, j] <= q95))
        oracle_width90 = float(np.mean(ow))
        oracle_cov90 = float(np.mean(ocov))
    else:
        z90 = float(_norm.ppf(0.95))
        oracle_width90 = float(np.mean(2.0 * z90 * osd))
        oracle_cov90 = float(np.mean(np.abs((y_true - om) / osd) <= z90))
    fi = model.fit_info
    if not (fi is not None):
        raise ValueError("fi is not None")
    return {
        "synthetic_model_crps": model_crps,
        "synthetic_model_mixture_crps": mix_crps,
        "synthetic_oracle_crps": oracle_crps,
        "synthetic_baseline_crps": baseline_crps,
        "synthetic_crps_gap_vs_oracle": model_crps - oracle_crps,
        "synthetic_crps_gain_vs_baseline": baseline_crps - model_crps,
        "synthetic_coverage_50": cov50,
        "synthetic_coverage_90": cov90,
        "synthetic_width_90": width90,
        "synthetic_oracle_coverage_90": oracle_cov90,
        "synthetic_oracle_width_90": oracle_width90,
        "synthetic_pit_ks": float(pit_stat),
        "synthetic_pit_ks_pvalue": float(pit_p),
        "synthetic_elbo_start": float(-fi.loss_curve[0]),
        "synthetic_final_elbo": float(fi.final_elbo),
        "synthetic_elbo_gain": float(-fi.loss_curve[-1] + fi.loss_curve[0]),
        "synthetic_final_kl_path": float(fi.final_kl_path),
        "synthetic_n_train": float(n_train),
        "synthetic_n_eval": float(n_eval),
        "synthetic_context_len": float(context_len),
        "synthetic_horizon": float(horizon),
        "synthetic_seed": float(seed),
        "synthetic_dgp": str(dgp),
        "synthetic_synthetic": "synthetic_known_law_sde_seeded",
        "synthetic_claim": "research_metric_only",
    }
