"""TORF: two-stage odd residual flows for mean-preserving probabilistic forecasts.

Madhusudhanan, K., Klötergens, C., Schmidt-Thieme, L. & Yalavarthi, V.K.
(2026), "Two-stage Odd Residual Flows for Mean-Preserving Probabilistic Time
Series Forecasting", arXiv:2608.11114 (cs.LG). Stage 1 is any deterministic
mean forecaster trained under a mean-focused loss and then FROZEN (the paper
uses SimpleTM; §4 "TORF is model-agnostic at this stage"). Stage 2 is a
Restricted Normalizing Flow (RNF; Kobayashi, T. & Aotani, T. 2023, "Design of
restricted normalizing flow towards arbitrary stochastic policy with
computational efficiency", Advanced Robotics 37:719-736) built from STRICTLY
ODD maps that learns the residual density around the point forecast:

- ROSS, Residual Odd Splines (Eqs. 7-9): ``v = sgn(u) * S(|u|; phi)`` where
  ``S`` is a monotone rational-quadratic spline (Durkan, C., Bekasov, A.,
  Murray, I. & Papamakarios, G. 2019, "Neural Spline Flows", NeurIPS 32) on
  ``[0, B]``, identity outside, parameterized by positive bin widths /
  heights / interior knot derivatives ``phi = (w_b, h_b, d_b)`` (``3*Nb - 1``
  free parameters; endpoint derivatives pinned to 1). Only the magnitude goes
  through ``S`` and the sign is restored afterwards, so oddness holds exactly
  by construction (``sgn(0) := +1`` convention keeps the map continuous).
- LSL, Linear Scaling Layer (Eqs. 11-12): ``v = u * s`` with a conditional
  scale ``s > 0`` from a bounded ScaleNet ``log s = a * tanh(MLP(F) / a)``
  (Eq. 13). Scaling ONLY: a translation would break oddness and decouple the
  predictive mean from Stage 1 (§4).
- Architecture (§4, Algorithm 1): ``K`` blocks ``[LSL -> ROSS]`` plus a
  trailing LSL; the LSL scale is shared by all ``K + 1`` scaling layers and
  each block has its own SplineNet. Training minimizes the NLL via the
  change of variables (Eq. 14) with a standard-normal latent.

Mean preservation (Lemma 1, Appendix B): a composition of odd maps is odd,
and an odd bijection pushes the symmetric N(0, I) base to a density symmetric
about zero, so ``E[eps | X] = 0`` and ``median(eps | X) = 0`` EXACTLY, with
no sampling. The predictive ``p(Y | X) = p_eps(Y - f_stage1(X) | X)`` (Eq. 5)
therefore keeps the frozen Stage-1 forecast as its exact analytical mean and
median -- faithful heteroscedastic regression in the sense of Stirn, A. et al.
(2023, AISTATS, PMLR 206:4653-4673), avoiding the mean-variance conflict of
joint NLL training (Seitzer, M. et al. 2022, ICLR, "On the Pitfalls of
Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks").
``K = 0`` reduces the flow exactly to a zero-mean conditional Gaussian
(Appendix G; the paper's MVE-2S baseline), and the output layer is
zero-initialized so training STARTS at that Gaussian model.

Deviations from the paper (documented, harness-driven):

1. Univariate marginal: one scalar target per observation -- the ``C = H = 1``
   case of the paper's conditional-independence assumption (§4 Stage 2) --
   conditioned on tabular context features instead of a lookback window.
2. SplineNet/ScaleNet are MLPs over the context, not Conv1D over a time axis.
   The paper's own ablation (Table 5, "w/o CNN-ROSS") finds the MLP equally
   accurate; the CNN exists for memory scaling on long multivariate horizons.
   The context embedding ``F`` is the standardized context itself (the
   paper's option (i) for non-differentiable Stage-1 models, §4 Context
   Embedding); the MLP hidden layers play the encoder role.
3. CRPS: the paper estimates CRPS from ``M = 100`` flow samples with the
   empirical CDF (Appendix F: "Neither has a closed form"). Here the flow CDF
   IS analytic -- ``F(v) = Phi(f(v / s_r))`` because the forward map ``f`` is
   strictly increasing -- so the CDF integral form (Gneiting, T. & Raftery,
   A.E. 2007, JASA 102:359-378, eq. 3; the paper's Eq. 21)

       CRPS(F, y) = int_{-inf}^{y} F(v)^2 dv + int_{y}^{inf} (1 - F(v))^2 dv

   is evaluated by deterministic composite Gauss-Legendre quadrature in the
   equivalent quantile (pinball) representation pushed into the flow's own
   latent variable -- bounded, spike-free, and scale-invariant (see
   :meth:`OddResidualFlow.crps`): no sampling error enters the score.
4. Stage-1 baseline: :class:`RidgeMeanForecaster`, a seeded closed-form ridge
   regression (the repo's theta/dlinear estimators consume series windows and
   do not import into the ``(X, y)`` harness contract). Any object exposing
   ``fit(X, y)`` / ``predict(X)`` can be injected into
   :class:`TORFForecaster` instead, matching the paper's model-agnosticism.
5. Optimizer: plain full-batch Adam (repo convention, cf. ``deep_hedging``)
   instead of the paper's tuned AdamW; learning rate defaults are harness
   choices, the paper tunes per dataset (Appendix N: K in {0,1,2,4,8},
   Nb in {4,8,16,32}).

Honesty (AGENTS.md contract): everything runs on SYNTHETIC seeded streams --
correctness evidence, never market evidence. Scores are proper (CRPS, log
score, PIT) plus point MAE of the frozen mean forecaster; no
Sharpe/Sortino/Calmar/P&L headline and no live-trading claims.

Conventions: torch is the optional ``nn`` extra, imported lazily via
:func:`_torch` (mirrors ``models/deep_hedging.py``), so this module imports
cleanly without torch; only training raises ``ImportError`` with install
guidance. ALL inference (forward/inverse flow maps, CDF, quantiles, log
density, CRPS quadrature, sampling) runs in numpy on float64 parameters
extracted after training. Training is CPU single-thread full-batch Adam,
deterministic given ``seed`` (GPU determinism is not claimed). Fail-closed:
invalid hyperparameters, non-finite or mismatched inputs, too-few samples,
zero-variance residuals, unfitted use, and non-finite training loss raise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf
from scipy.stats import norm

Array = NDArray[np.float64]

__all__ = [
    "OddFlowFitInfo",
    "OddResidualFlow",
    "RidgeMeanForecaster",
    "TORFForecaster",
    "bench_odd_residual_flows",
]

_LOG_2PI = math.log(2.0 * math.pi)
# Quadrature defaults for the CRPS CDF integral (see OddResidualFlow.crps).
_CRPS_PANELS = 40
_CRPS_NODES = 20
_CRPS_TAIL_TAU = 1e-10
# Effective latent clamp for the CRPS arms: phi(10) < 1e-22, so integrand mass
# beyond |z| = 10 is negligible even when an extreme outlier pushes the split
# point z_y far into the Gaussian range (see OddResidualFlow.crps).
_CRPS_LATENT_CLAMP = 10.0
# Spline positivity floors (Durkan et al. 2019 reference defaults are 1e-3
# scale; the derivative floor also keeps the rational branches well-conditioned
# for the CRPS quadrature — near-zero derivatives put rational-function poles
# close to the real axis and degrade Gauss-Legendre convergence).
_BIN_MASS_FLOOR = 1e-4  # minimum softmax mass per bin (widths/heights)
_DERIV_FLOOR = 1e-3  # added to softplus'd interior knot derivatives
_MIN_FIT_SAMPLES = 8
# softplus(x) = 1 - _DERIV_FLOOR  =>  interior derivative exactly 1 at init.
_SOFTPLUS_INV_UNIT = math.log(math.expm1(1.0 - _DERIV_FLOOR))

_Layers = tuple[tuple[Array, Array], ...]


def _torch() -> Any:
    """Lazily import torch (optional ``nn`` extra); fail closed with guidance."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "odd residual flows need the optional 'nn' extra (torch): uv sync --extra nn"
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
# numpy core: rational-quadratic splines and the odd ROSS extension
# ---------------------------------------------------------------------------


def _softmax_floor(raw: Array, n_bins: int) -> Array:
    """Softmax over the last axis with a per-bin minimum mass floor."""
    z = raw - np.max(raw, axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / np.sum(e, axis=-1, keepdims=True)
    m = _BIN_MASS_FLOOR
    return np.asarray(p * (1.0 - n_bins * m) + m, dtype=float)


def _spline_params_from_raw(raw: Array, n_bins: int, bound: float) -> tuple[Array, Array, Array]:
    """Unconstrained raw ``(..., 3*Nb-1)`` -> positive ``(w, h, d)``.

    Widths/heights are floored-softmax masses scaled to sum to ``bound``;
    interior derivatives are ``softplus + floor`` and the two endpoint
    derivatives are pinned to 1 so the spline meets the identity tail with a
    continuous first derivative (Durkan et al. 2019; the paper's ``3*Nb-1``
    free parameters, §3 RQ-NSF).
    """
    raw_w = raw[..., :n_bins]
    raw_h = raw[..., n_bins : 2 * n_bins]
    raw_d = raw[..., 2 * n_bins :]
    w = bound * _softmax_floor(raw_w, n_bins)
    h = bound * _softmax_floor(raw_h, n_bins)
    d_inner = np.logaddexp(0.0, raw_d) + _DERIV_FLOOR
    ones = np.ones(d_inner.shape[:-1] + (1,))
    d = np.asarray(np.concatenate([ones, d_inner, ones], axis=-1), dtype=float)
    return np.asarray(w, dtype=float), np.asarray(h, dtype=float), d


def _expand_bins(arr: Array, x: NDArray[Any]) -> Array:
    """Reshape a per-bin parameter array so it broadcasts against point array ``x``.

    The parameter leading dims (everything but the bin axis) align with the
    LEADING dims of ``x``; ``x``'s trailing point axes (samples, quadrature
    panels/nodes) are padded with singleton dims so they broadcast over bins.
    Inserting singleton axes is a view -- no copy.
    """
    n_lead = arr.ndim - 1
    pad = x.ndim - n_lead
    if pad < 0:
        raise ValueError("bin parameters have more leading dims than the evaluation points")
    return arr.reshape(arr.shape[:n_lead] + (1,) * pad + (arr.shape[-1],))


def _bin_index(x: Array, cum: Array) -> NDArray[np.intp]:
    """Bin index of ``x`` against knot positions ``cum`` (last axis, Nb+1)."""
    cum_r = _expand_bins(cum, x)
    idx = np.sum(x[..., None] >= cum_r[..., 1:], axis=-1)
    return np.clip(idx, 0, cum.shape[-1] - 2).astype(np.intp)


def _gather_bins(arr: Array, idx: NDArray[np.intp]) -> Array:
    """Gather per-bin values ``arr[..., idx]`` with leading-dim broadcasting."""
    arr_r = _expand_bins(arr, idx)
    full = np.broadcast_shapes(arr_r.shape[:-1], idx.shape)
    arr_b = np.broadcast_to(arr_r, full + (arr.shape[-1],))
    idx_b = np.broadcast_to(idx, full)
    return np.take_along_axis(arr_b, idx_b[..., None], axis=-1)[..., 0]


def _rq_spline_forward_mag(
    x: Array, w: Array, h: Array, d: Array, *, bound: float
) -> tuple[Array, Array]:
    """Monotone rational-quadratic spline ``S`` on ``[0, bound]``, identity outside.

    Durkan et al. (2019) forward map and derivative, evaluated on magnitudes
    (the odd extension lives in :func:`_ross_forward`). ``x`` broadcasts
    against the leading dims of ``w``/``h`` (``(..., Nb)``) and ``d``
    (``(..., Nb+1)``). Returns ``(S(x), S'(x))`` with ``S' = 1`` outside.
    """
    cumw = np.concatenate([np.zeros(w.shape[:-1] + (1,)), np.cumsum(w, axis=-1)], axis=-1)
    cumh = np.concatenate([np.zeros(h.shape[:-1] + (1,)), np.cumsum(h, axis=-1)], axis=-1)
    inside = x < bound
    xs = np.clip(x, 0.0, bound * (1.0 - 1e-7))
    idx = _bin_index(xs, cumw)
    w_i = _gather_bins(w, idx)
    h_i = _gather_bins(h, idx)
    d_i = _gather_bins(d, idx)
    d_i1 = _gather_bins(d, idx + 1)
    cw_i = _gather_bins(cumw, idx)
    ch_i = _gather_bins(cumh, idx)
    xi = np.clip((xs - cw_i) / w_i, 0.0, 1.0)
    om = 1.0 - xi
    s_i = h_i / w_i
    denom = s_i + (d_i1 + d_i - 2.0 * s_i) * xi * om
    y = ch_i + h_i * (s_i * xi * xi + d_i * xi * om) / denom
    deriv = (s_i * s_i) * (d_i1 * xi * xi + 2.0 * s_i * xi * om + d_i * om * om) / (denom * denom)
    return np.asarray(np.where(inside, y, x), dtype=float), np.asarray(
        np.where(inside, deriv, 1.0), dtype=float
    )


def _rq_spline_inverse_mag(y: Array, w: Array, h: Array, d: Array, *, bound: float) -> Array:
    """Analytic inverse of :func:`_rq_spline_forward_mag` (Durkan et al. 2019)."""
    cumw = np.concatenate([np.zeros(w.shape[:-1] + (1,)), np.cumsum(w, axis=-1)], axis=-1)
    cumh = np.concatenate([np.zeros(h.shape[:-1] + (1,)), np.cumsum(h, axis=-1)], axis=-1)
    inside = y < bound
    ys = np.clip(y, 0.0, bound * (1.0 - 1e-7))
    idx = _bin_index(ys, cumh)
    w_i = _gather_bins(w, idx)
    h_i = _gather_bins(h, idx)
    d_i = _gather_bins(d, idx)
    d_i1 = _gather_bins(d, idx + 1)
    cw_i = _gather_bins(cumw, idx)
    ch_i = _gather_bins(cumh, idx)
    s_i = h_i / w_i
    dy = ys - ch_i
    a = dy * (d_i1 + d_i - 2.0 * s_i) + h_i * (s_i - d_i)
    b = h_i * d_i - dy * (d_i1 + d_i - 2.0 * s_i)
    c = -s_i * dy
    disc = np.maximum(b * b - 4.0 * a * c, 0.0)
    denom = -b - np.sqrt(disc)
    xi = np.where(denom == 0.0, 0.0, 2.0 * c / np.where(denom == 0.0, 1.0, denom))
    xi = np.clip(xi, 0.0, 1.0)
    x = cw_i + w_i * xi
    return np.asarray(np.where(inside, x, y), dtype=float)


def _ross_forward(u: Array, w: Array, h: Array, d: Array, *, bound: float) -> tuple[Array, Array]:
    """ROSS layer (paper Eqs. 7/9): ``v = sgn(u) * S(|u|)``, ``ldj = log S'(|u|)``.

    Odd by construction; ``sgn(0) := +1`` (the paper's convention) keeps the
    map continuous at the origin because ``S(0) = 0``. The log-derivative is
    an even function of ``u``.
    """
    sign = np.where(u >= 0.0, 1.0, -1.0)
    y, deriv = _rq_spline_forward_mag(np.abs(u), w, h, d, bound=bound)
    return np.asarray(sign * y, dtype=float), np.asarray(np.log(deriv), dtype=float)


def _ross_inverse(v: Array, w: Array, h: Array, d: Array, *, bound: float) -> Array:
    """Inverse ROSS layer (paper Eq. 8): ``u = sgn(v) * S^{-1}(|v|)``."""
    sign = np.where(v >= 0.0, 1.0, -1.0)
    return np.asarray(sign * _rq_spline_inverse_mag(np.abs(v), w, h, d, bound=bound), dtype=float)


def _np_mlp(F: Array, layers: Sequence[tuple[Array, Array]]) -> Array:
    """Numpy mirror of the training MLPs: tanh hidden layers, linear output."""
    z = F
    last = len(layers) - 1
    for i, (W, b) in enumerate(layers):
        z = z @ W.T + b
        if i < last:
            z = np.tanh(z)
    return np.asarray(z, dtype=float)


@dataclass(frozen=True, eq=False)
class _NumpyFlowParams:
    """Float64 inference parameters extracted from the torch training run."""

    n_blocks: int
    n_bins: int
    tail_bound: float
    log_scale_bound: float
    resid_std: float
    ctx_mean: Array
    ctx_std: Array
    scale_layers: _Layers
    spline_layers: tuple[_Layers, ...]


def _np_flow_context(
    params: _NumpyFlowParams, C: Array
) -> tuple[Array, list[tuple[Array, Array, Array]]]:
    """Context -> (log scale, per-block spline params), all numpy."""
    F = (C - params.ctx_mean) / params.ctx_std
    a = params.log_scale_bound
    log_s = a * np.tanh(_np_mlp(F, params.scale_layers)[..., 0] / a)
    spline_list = [
        _spline_params_from_raw(_np_mlp(F, layers), params.n_bins, params.tail_bound)
        for layers in params.spline_layers
    ]
    return np.asarray(log_s, dtype=float), spline_list


def _np_forward_std(
    params: _NumpyFlowParams,
    log_s: Array,
    spline_list: Sequence[tuple[Array, Array, Array]],
    r_std: Array,
) -> tuple[Array, Array]:
    """Forward flow map on standardized residuals (Algorithm 1, forward pass).

    ``LSL -> [ROSS -> LSL] * K``: one scaling before the first spline, one
    after every spline (the shared scale implements all ``K + 1`` LSLs).
    Point arrays broadcast against context arrays on trailing dims, so the
    same routine serves vectors, sample grids, and CRPS quadrature panels.
    Returns ``(latent z, forward log-det-Jacobian)``.
    """
    s = np.exp(log_s)
    z = r_std * s
    ldj = np.zeros_like(z) + log_s
    for w, h, d in spline_list:
        z, ldj_r = _ross_forward(z, w, h, d, bound=params.tail_bound)
        ldj = ldj + ldj_r
        z = z * s
        ldj = ldj + log_s
    return np.asarray(z, dtype=float), np.asarray(ldj, dtype=float)


def _np_inverse_std(
    params: _NumpyFlowParams,
    log_s: Array,
    spline_list: Sequence[tuple[Array, Array, Array]],
    zl: Array,
) -> Array:
    """Inverse flow map (paper Appendix D: inference runs the inverse pass)."""
    s = np.exp(log_s)
    z = zl / s
    for k in range(params.n_blocks - 1, -1, -1):
        w, h, d = spline_list[k]
        z = _ross_inverse(z, w, h, d, bound=params.tail_bound)
        z = z / s
    return np.asarray(z, dtype=float)


def _latent_knot_positions(
    params: _NumpyFlowParams, log_s: Array, spline_list: Sequence[tuple[Array, Array, Array]]
) -> Array:
    """Latent images of every ROSS bin boundary: the ONLY non-C^2 points of Q.

    The quantile map ``Q(z) = f^{-1}(z)`` is piecewise analytic with C^1 kinks
    (second-derivative jumps) exactly at ``z = f(v_knot)``, where ``v_knot``
    runs over each block's spline bin boundaries in standardized residual
    space: the origin, the interior width knots ``+-c_j``, and the ``+-B``
    identity joins. Each block-k knot value ``kappa`` (in that block's ROSS
    input space) is pulled back through the preceding blocks to residual
    space and pushed through the full forward map. Returns ``(n, K(2Nb+1))``
    (empty for K = 0, where Q is analytic). Used by the CRPS quadrature to
    place panel edges ON the kinks, restoring spectral convergence per panel.
    """
    n = int(log_s.shape[0])
    if params.n_blocks == 0:
        return np.zeros((n, 0))
    s = np.exp(log_s)[:, None]
    nb = params.n_bins
    bound = params.tail_bound
    out: list[Array] = []
    for k, (w, _h, _d) in enumerate(spline_list):
        cumw = np.cumsum(w, axis=-1)  # c_1 .. c_Nb (c_Nb = bound up to float)
        interior = cumw[:, : max(nb - 1, 0)]
        kappa = np.concatenate(
            [
                np.zeros((n, 1)),
                interior,
                -interior,
                np.full((n, 1), bound),
                np.full((n, 1), -bound),
            ],
            axis=1,
        )  # (n, 2Nb+1) knots in block k's ROSS input space U^(k+1/2)
        u = kappa / s  # undo block k's LSL
        for j in range(k - 1, -1, -1):
            wj, hj, dj = spline_list[j]
            u = _ross_inverse(u, wj, hj, dj, bound=bound)
            u = u / s
        z, _ = _np_forward_std(params, log_s[:, None], spline_list, u)
        out.append(np.asarray(z, dtype=float))
    return np.concatenate(out, axis=1)


# ---------------------------------------------------------------------------
# torch training core (guarded; mirrors the numpy inference math exactly)
# ---------------------------------------------------------------------------


def _build_mlp(torch: Any, d_in: int, hidden: Sequence[int], d_out: int) -> Any:
    layers: list[Any] = []
    d = int(d_in)
    for hh in hidden:
        layers += [torch.nn.Linear(d, int(hh)), torch.nn.Tanh()]
        d = int(hh)
    layers.append(torch.nn.Linear(d, int(d_out)))
    return torch.nn.Sequential(*layers)


def _init_zero_last(torch: Any, net: Any, *, n_bins: int) -> None:
    """Zero the output layer so the flow STARTS as the exact conditional Gaussian.

    ScaleNet -> ``log s = 0`` (unit scale); SplineNet -> uniform bins and
    interior knot derivatives of exactly 1, which is the identity spline. At
    initialization the K-block flow therefore IS the K = 0 Gaussian model
    (arXiv:2608.11114 Appendix G), and training departs from the MVE-2S
    baseline only where the NLL pays for it.
    """
    last = net[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
        if n_bins > 0:
            last.bias[2 * n_bins :] = _SOFTPLUS_INV_UNIT


def _torch_softmax_floor(torch: Any, raw: Any, n_bins: int) -> Any:
    p = torch.softmax(raw, dim=-1)
    m = _BIN_MASS_FLOOR
    return p * (1.0 - n_bins * m) + m


def _torch_spline_params(torch: Any, raw: Any, n_bins: int, bound: float) -> tuple[Any, Any, Any]:
    raw_w = raw[..., :n_bins]
    raw_h = raw[..., n_bins : 2 * n_bins]
    raw_d = raw[..., 2 * n_bins :]
    w = bound * _torch_softmax_floor(torch, raw_w, n_bins)
    h = bound * _torch_softmax_floor(torch, raw_h, n_bins)
    d_inner = torch.nn.functional.softplus(raw_d) + _DERIV_FLOOR
    ones = torch.ones_like(d_inner[..., :1])
    d = torch.cat([ones, d_inner, ones], dim=-1)
    return w, h, d


def _torch_ross_forward(
    torch: Any, u: Any, w: Any, h: Any, d: Any, *, bound: float
) -> tuple[Any, Any]:
    """Differentiable ROSS layer; magnitudes are clamped inside the spline so
    the identity-tail branch of ``torch.where`` never sees a NaN gradient."""
    sign = torch.where(u >= 0.0, 1.0, -1.0)
    mag = u.abs()
    inside = mag < bound
    mag_s = torch.clamp(mag, max=bound * (1.0 - 1e-7))
    cumw = torch.cat([torch.zeros_like(w[..., :1]), torch.cumsum(w, dim=-1)], dim=-1)
    cumh = torch.cat([torch.zeros_like(h[..., :1]), torch.cumsum(h, dim=-1)], dim=-1)
    n_bins = int(w.shape[-1])
    idx = torch.sum((mag_s.unsqueeze(-1) >= cumw[..., 1:]).to(torch.int64), dim=-1)
    idx = torch.clamp(idx, 0, n_bins - 1)
    ix = idx.unsqueeze(-1)
    w_i = torch.gather(w, -1, ix).squeeze(-1)
    h_i = torch.gather(h, -1, ix).squeeze(-1)
    d_i = torch.gather(d, -1, ix).squeeze(-1)
    d_i1 = torch.gather(d, -1, ix + 1).squeeze(-1)
    cw_i = torch.gather(cumw, -1, ix).squeeze(-1)
    ch_i = torch.gather(cumh, -1, ix).squeeze(-1)
    xi = torch.clamp((mag_s - cw_i) / w_i, 0.0, 1.0)
    om = 1.0 - xi
    s_i = h_i / w_i
    denom = s_i + (d_i1 + d_i - 2.0 * s_i) * xi * om
    y = ch_i + h_i * (s_i * xi * xi + d_i * xi * om) / denom
    deriv = (s_i * s_i) * (d_i1 * xi * xi + 2.0 * s_i * xi * om + d_i * om * om) / (denom * denom)
    y = torch.where(inside, y, mag)
    deriv = torch.where(inside, deriv, torch.ones_like(deriv))
    return sign * y, torch.log(deriv)


def _torch_flow_forward(
    torch: Any,
    F_t: Any,
    r_t: Any,
    scale_net: Any,
    spline_nets: Sequence[Any],
    *,
    a: float,
    n_bins: int,
    bound: float,
) -> tuple[Any, Any]:
    """Algorithm 1 forward pass: residual -> standard-normal latent + ldj."""
    log_s = a * torch.tanh(scale_net(F_t).squeeze(-1) / a)
    s = torch.exp(log_s)
    z = r_t * s
    ldj = torch.zeros_like(z) + log_s
    for net in spline_nets:
        w, h, d = _torch_spline_params(torch, net(F_t), n_bins, bound)
        z, ldj_r = _torch_ross_forward(torch, z, w, h, d, bound=bound)
        ldj = ldj + ldj_r
        z = z * s
        ldj = ldj + log_s
    return z, ldj


def _train_flow(
    torch: Any,
    Cs: Array,
    r_std: Array,
    *,
    n_blocks: int,
    n_bins: int,
    bound: float,
    hidden: Sequence[int],
    log_scale_bound: float,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[Any, list[Any], list[float]]:
    """Full-batch Adam on the flow NLL (Eq. 14); deterministic given seed."""
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    p = int(Cs.shape[1])
    scale_net = _build_mlp(torch, p, hidden, 1)
    _init_zero_last(torch, scale_net, n_bins=0)
    spline_nets = [_build_mlp(torch, p, hidden, 3 * n_bins - 1) for _ in range(int(n_blocks))]
    for net in spline_nets:
        _init_zero_last(torch, net, n_bins=int(n_bins))
    params: list[Any] = list(scale_net.parameters())
    for net in spline_nets:
        params.extend(net.parameters())
    opt = torch.optim.Adam(params, lr=float(lr))
    F_t = torch.as_tensor(Cs, dtype=torch.float32)
    r_t = torch.as_tensor(r_std, dtype=torch.float32)
    curve: list[float] = []
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        z, ldj = _torch_flow_forward(
            torch,
            F_t,
            r_t,
            scale_net,
            spline_nets,
            a=float(log_scale_bound),
            n_bins=int(n_bins),
            bound=float(bound),
        )
        nll = torch.mean(0.5 * z * z + 0.5 * _LOG_2PI - ldj)
        val = float(nll.detach().numpy())
        if not math.isfinite(val):
            raise ValueError("flow NLL is not finite during training")
        curve.append(val)
        nll.backward()
        opt.step()
    return scale_net, spline_nets, curve


def _extract_mlp_layers(torch: Any, net: Any) -> _Layers:
    out: list[tuple[Array, Array]] = []
    for module in net:
        if isinstance(module, torch.nn.Linear):
            W = np.array(module.weight.detach().numpy(), dtype=float)
            b = np.array(module.bias.detach().numpy(), dtype=float)
            out.append((W, b))
    return tuple(out)


# ---------------------------------------------------------------------------
# Stage 2: the odd residual flow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OddFlowFitInfo:
    """Stage-2 training trace.

    ``loss_curve`` holds the standardized-space flow NLL (Eq. 14) recorded
    BEFORE each Adam update, so entry 0 is the loss at the exact conditional
    Gaussian initialization (identity splines, unit scales). ``final_loss``
    is the NLL of the RAW residuals, recomputed on the training data through
    the extracted numpy inference path; it exceeds the standardized objective
    by exactly ``log(resid_std)``.
    """

    epochs: int
    loss_curve: list[float]
    final_loss: float
    seed: int
    n_blocks: int
    n_bins: int


class OddResidualFlow:
    """Stage-2 residual density: a restricted normalizing flow of odd maps.

    Models ``p(eps | context)`` with ``E[eps | context] = 0`` and coordinate
    median 0 BY CONSTRUCTION (arXiv:2608.11114, Lemma 1): every layer (LSL
    scaling, ROSS sign-restored spline) is strictly odd, so the pushforward
    of the symmetric standard-normal base is symmetric about zero -- no
    sampling needed for the mean. ``fit`` needs torch (the ``nn`` extra);
    every inference method is pure numpy on the extracted float64 parameters.
    Residuals are standardized by their training std internally, so the
    spline tail bound ``B`` is in residual-std units.
    """

    def __init__(
        self,
        n_blocks: int = 2,
        n_bins: int = 8,
        tail_bound: float = 5.0,
        hidden: Sequence[int] = (32, 32),
        log_scale_bound: float = 2.5,
        epochs: int = 300,
        lr: float = 8e-3,
        seed: int = 0,
    ) -> None:
        if isinstance(n_blocks, bool) or int(n_blocks) != n_blocks or int(n_blocks) < 0:
            raise ValueError(
                f"n_blocks must be an int >= 0 (K = 0 is the Gaussian case); got {n_blocks!r}"
            )
        if isinstance(n_bins, bool) or int(n_bins) != n_bins or int(n_bins) < 1:
            raise ValueError(f"n_bins must be an int >= 1; got {n_bins!r}")
        hidden_widths = tuple(int(hh) for hh in hidden)
        if not hidden_widths or any(hh < 1 for hh in hidden_widths):
            raise ValueError(
                f"hidden must be a non-empty sequence of positive widths; got {hidden!r}"
            )
        if isinstance(epochs, bool) or int(epochs) != epochs or int(epochs) < 1:
            raise ValueError(f"epochs must be an int >= 1; got {epochs!r}")
        self.n_blocks = int(n_blocks)
        self.n_bins = int(n_bins)
        self.tail_bound = _check_positive(tail_bound, "tail_bound")
        self.hidden = hidden_widths
        self.log_scale_bound = _check_positive(log_scale_bound, "log_scale_bound")
        self.epochs = int(epochs)
        self.lr = _check_positive(lr, "lr")
        self.seed = int(seed)
        self._params: _NumpyFlowParams | None = None
        self.fit_info: OddFlowFitInfo | None = None

    # -- state --------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._params is not None

    @property
    def resid_std(self) -> float:
        """Training residual std used for internal standardization."""
        return float(self._require_params().resid_std)

    @property
    def n_features(self) -> int:
        return int(self._require_params().ctx_mean.size)

    def _require_params(self) -> _NumpyFlowParams:
        if self._params is None:
            raise RuntimeError("OddResidualFlow is not fitted")
        return self._params

    def _context(self, C: Array) -> Array:
        params = self._require_params()
        arr = _check_matrix(C, "C")
        if arr.shape[1] != params.ctx_mean.size:
            raise ValueError(
                f"context feature count mismatch: fitted with {params.ctx_mean.size}, "
                f"got {arr.shape[1]}"
            )
        return arr

    # -- training (torch-gated) ---------------------------------------------

    def fit(self, C: Array, r: Array) -> OddResidualFlow:
        """Fit the odd flow to Stage-1 residuals ``r`` given context ``C``.

        Inputs are validated BEFORE torch is touched, so input-contract errors
        raise ``ValueError`` even without the ``nn`` extra; a successful fit
        needs torch and raises ``ImportError`` with guidance when absent.
        """
        Cm = _check_matrix(C, "C")
        rv = _check_vector(r, Cm.shape[0], "r")
        if Cm.shape[0] < _MIN_FIT_SAMPLES:
            raise ValueError(f"too few samples: need >= {_MIN_FIT_SAMPLES}, got {Cm.shape[0]}")
        resid_std = float(np.std(rv))
        if not math.isfinite(resid_std) or resid_std <= 0.0:
            raise ValueError("residuals have zero variance; nothing to fit")
        ctx_mean = Cm.mean(axis=0)
        ctx_std = Cm.std(axis=0)
        ctx_std = np.where(ctx_std > 0.0, ctx_std, 1.0)  # constant columns: unit scale
        Cs = np.asarray((Cm - ctx_mean) / ctx_std, dtype=float)
        r_std = np.asarray(rv / resid_std, dtype=float)

        torch = _torch()
        scale_net, spline_nets, curve = _train_flow(
            torch,
            Cs,
            r_std,
            n_blocks=self.n_blocks,
            n_bins=self.n_bins,
            bound=self.tail_bound,
            hidden=self.hidden,
            log_scale_bound=self.log_scale_bound,
            epochs=self.epochs,
            lr=self.lr,
            seed=self.seed,
        )
        self._params = _NumpyFlowParams(
            n_blocks=self.n_blocks,
            n_bins=self.n_bins,
            tail_bound=self.tail_bound,
            log_scale_bound=self.log_scale_bound,
            resid_std=resid_std,
            ctx_mean=np.asarray(ctx_mean, dtype=float),
            ctx_std=np.asarray(ctx_std, dtype=float),
            scale_layers=_extract_mlp_layers(torch, scale_net),
            spline_layers=tuple(_extract_mlp_layers(torch, net) for net in spline_nets),
        )
        raw_nll = float(np.mean(-self.log_prob(Cm, rv)))
        if not math.isfinite(raw_nll):
            self._params = None
            raise ValueError("extracted flow parameters produce a non-finite NLL")
        self.fit_info = OddFlowFitInfo(
            epochs=self.epochs,
            loss_curve=curve,
            final_loss=raw_nll,
            seed=self.seed,
            n_blocks=self.n_blocks,
            n_bins=self.n_bins,
        )
        return self

    # -- numpy inference ------------------------------------------------------

    def forward(self, C: Array, r: Array) -> tuple[Array, Array]:
        """Forward (data -> latent) map: ``z = f(r / s_r)``, plus log-det-Jacobian.

        The ldj includes the fixed standardization scale, so
        ``log_prob = log N(z) + ldj`` is the density of the RAW residual.
        """
        params = self._require_params()
        Cx = self._context(C)
        rv = _check_vector(r, Cx.shape[0], "r")
        log_s, spline_list = _np_flow_context(params, Cx)
        z, ldj = _np_forward_std(params, log_s, spline_list, rv / params.resid_std)
        return z, np.asarray(ldj - math.log(params.resid_std), dtype=float)

    def inverse(self, C: Array, zl: Array) -> Array:
        """Inverse (latent -> data) map: raw residuals for latent draws ``zl``."""
        params = self._require_params()
        Cx = self._context(C)
        zv = _check_vector(zl, Cx.shape[0], "zl")
        log_s, spline_list = _np_flow_context(params, Cx)
        r_std = _np_inverse_std(params, log_s, spline_list, zv)
        return np.asarray(r_std * params.resid_std, dtype=float)

    def log_prob(self, C: Array, r: Array) -> Array:
        """Log predictive density of residuals (proper log score ingredient)."""
        z, ldj = self.forward(C, r)
        return np.asarray(-0.5 * z * z - 0.5 * _LOG_2PI + ldj, dtype=float)

    def cdf(self, C: Array, v: Array) -> Array:
        """Analytic residual CDF ``F(v) = Phi(f(v / s_r))`` per observation.

        Exact (no sampling) because the odd flow map ``f`` is strictly
        increasing: ``P(eps <= v) = P(Z <= f(v / s_r))`` with ``Z ~ N(0, 1)``.
        """
        params = self._require_params()
        Cx = self._context(C)
        vv = _check_vector(v, Cx.shape[0], "v")
        log_s, spline_list = _np_flow_context(params, Cx)
        z, _ = _np_forward_std(params, log_s, spline_list, vv / params.resid_std)
        return np.asarray(0.5 * (1.0 + erf(z / math.sqrt(2.0))), dtype=float)

    def quantile(self, C: Array, taus: Array) -> Array:
        """Residual quantiles ``Q(tau) = s_r * f^{-1}(Phi^{-1}(tau))``, shape (n, T).

        ``Q(0.5) = 0`` EXACTLY (odd map: ``f(0) = 0``): the paper's zero-median
        guarantee (Lemma 1), sampling-free.
        """
        params = self._require_params()
        Cx = self._context(C)
        t = np.asarray(taus, dtype=float).ravel()
        if t.size == 0 or not bool(np.all(np.isfinite(t))) or np.any(t <= 0.0) or np.any(t >= 1.0):
            raise ValueError(f"taus must be non-empty, finite, and in (0, 1); got {taus!r}")
        n = Cx.shape[0]
        zq = np.asarray(norm.ppf(t), dtype=float)
        log_s, spline_list = _np_flow_context(params, Cx)
        r_std = _np_inverse_std(
            params, log_s[:, None], spline_list, np.broadcast_to(zq[None, :], (n, t.size))
        )
        return np.asarray(r_std * params.resid_std, dtype=float)

    def sample(self, C: Array, n_samples: int, seed: int = 0) -> Array:
        """Seeded residual draws via the inverse pass (paper Appendix D), (n, M)."""
        params = self._require_params()
        Cx = self._context(C)
        m = _check_count(n_samples, "n_samples")
        rng = np.random.default_rng(int(seed))
        zl = rng.standard_normal((Cx.shape[0], m))
        log_s, spline_list = _np_flow_context(params, Cx)
        r_std = _np_inverse_std(params, log_s[:, None], spline_list, zl)
        return np.asarray(r_std * params.resid_std, dtype=float)

    def crps(
        self,
        C: Array,
        r: Array,
        *,
        n_panels: int = _CRPS_PANELS,
        n_nodes: int = _CRPS_NODES,
        tail_tau: float = _CRPS_TAIL_TAU,
    ) -> Array:
        """Elementwise CRPS of the residual density via CDF quadrature.

        CDF integral form (Gneiting & Raftery 2007, eq. 3; paper Eq. 21):

            CRPS(F, y) = int_{-inf}^{y} F(v)^2 dv
                       + int_{y}^{inf} (1 - F(v))^2 dv,   F(v) = Phi(f(v/s_r)).

        QUADRATURE (deterministic, no sampling -- unlike the paper's M-sample
        empirical-CDF estimator, Appendix F). We integrate the EQUIVALENT
        quantile (pinball) form of the same proper score (Gneiting & Raftery
        2007; the representation ``quant_fund.metrics.scoring.
        crps_from_quantiles`` discretizes), pushed into the flow's latent
        variable ``z`` via ``tau = Phi(z)``, ``d tau = phi(z) dz``:

            CRPS(F, y) = 2 int_0^1 rho_tau(y, Q(tau)) dtau
                       = 2 [ int_{-inf}^{z_y} Phi(z) (y - Q(z)) phi(z) dz
                           + int_{z_y}^{inf} (1 - Phi(z)) (Q(z) - y) phi(z) dz ],

        with flow quantile ``Q(z) = s_r * f^{-1}(z)``, split point
        ``z_y = f(y / s_r)`` (so ``Phi(z_y) = F(y)`` is the exact predictive
        PIT), and ``phi``/``Phi`` the standard-normal pdf/cdf. This latent
        parameterization is the well-conditioned choice: the integrand uses
        ``f^{-1}(z)`` (BOUNDED) rather than its derivative, times the rapidly
        decaying ``phi(z)``, so it never develops the ``exp(-ldj)`` spikes of
        the raw ``v``/``z`` CDF form nor the ``Phi^{-1}`` endpoint singularity
        of the ``tau`` form. It is scale-invariant (LSL/ROSS compression is
        absorbed by ``f^{-1}``) and both arms vanish as ``z -> +-inf``. Panel
        edges are placed EXACTLY on the two kinds of non-smooth points: the
        indicator split ``z_y`` and the latent knot images -- the only points
        where ``Q = f^{-1}`` is not C^2 (the rational-quadratic spline inverse
        is C^1 with second-derivative jumps at bin boundaries and at the
        ``+-B`` identity joins; ``_latent_knot_positions`` enumerates them) --
        merged with ``n_panels`` uniform scaffold cells per arm. Between edges
        the integrand is analytic, so ``n_nodes``-point Gauss-Legendre
        converges spectrally per panel. The semi-infinite arms are
        truncated at ``z = Phi^{-1}(tail_tau)`` / ``Phi^{-1}(1 - tail_tau)``
        (default 1e-10 -> ``|z| <= 6.4``), clamped outward to ``z_y`` so
        far-out outliers keep panel order; the dropped Gaussian tails are
        bounded by ``tail_tau * E|y - eps| = O(1e-10)`` -- far below any
        Monte-Carlo comparison tolerance. The split point is additionally
        clamped to ``|z_y| <= 10`` (``phi(10) < 1e-22``): for extreme outlier
        observations this keeps panels dense over the region that carries all
        integrand mass; the dropped piece is bounded by
        ``(|y| + |Q(10)|) * Phi(-10) = O(1e-22 * scale)``.

        Accuracy (validated on a trained heavy-tailed flow against a
        2,000,000-sample empirical CRPS, which the quadrature matched within
        the sampler's own MC noise): at the default 40 panels x 20 nodes the
        worst-observation absolute error is ~1e-6 and the test-set MEAN error
        ~1e-9; the exact Gaussian case (K = 0) reproduces
        ``crps_gaussian`` to ~1e-15.
        """
        params = self._require_params()
        n_pan = _check_count(n_panels, "n_panels")
        n_nod = _check_count(n_nodes, "n_nodes")
        if not 0.0 < float(tail_tau) < 0.5:
            raise ValueError(f"tail_tau must be in (0, 0.5); got {tail_tau!r}")
        Cx = self._context(C)
        rv = _check_vector(r, Cx.shape[0], "r")
        log_s, spline_list = _np_flow_context(params, Cx)
        z_y, _ = _np_forward_std(params, log_s, spline_list, rv / params.resid_std)
        z_lo = float(norm.ppf(float(tail_tau)))
        z_hi = float(norm.ppf(1.0 - float(tail_tau)))
        z_split = np.clip(z_y, -_CRPS_LATENT_CLAMP, _CRPS_LATENT_CLAMP)
        a = np.minimum(z_lo, z_split)
        b = np.maximum(z_hi, z_split)
        knots = _latent_knot_positions(params, log_s, spline_list)  # (n, nk) C^1 kinks of Q
        nodes, weights = np.polynomial.legendre.leggauss(n_nod)
        frac = np.linspace(0.0, 1.0, n_pan + 1)

        def _arm(lo: Array, hi: Array, upper: bool) -> Array:
            """Composite-GL integral of one latent-space pinball arm.

            Panel edges are the union of ``n_pan`` uniform cells, the arm
            bounds, and the knot images clipped into ``[lo, hi]`` -- so every
            C^1 kink of ``Q = f^{-1}`` sits ON an edge and each panel is
            analytic (spectral GL convergence). Coincident/clipped edges give
            zero-width panels that contribute exactly 0.
            """
            uniform = lo[:, None] + (hi - lo)[:, None] * frac[None, :]  # (n, n_pan+1)
            if knots.shape[1] > 0:
                knot_in = np.clip(knots, lo[:, None], hi[:, None])  # (n, nk)
                edges = np.sort(np.concatenate([uniform, knot_in], axis=1), axis=1)
            else:
                edges = uniform
            left = edges[:, :-1]
            width = edges[:, 1:] - left
            zz = left[:, :, None] + 0.5 * width[:, :, None] * (nodes[None, None, :] + 1.0)
            wts = 0.5 * width[:, :, None] * weights[None, None, :]
            q_raw = _np_inverse_std(params, log_s[:, None, None], spline_list, zz)
            q_raw = q_raw * params.resid_std
            phi = np.exp(-0.5 * zz * zz) / math.sqrt(2.0 * math.pi)
            cdf = 0.5 * (1.0 + erf(zz / math.sqrt(2.0)))
            y_b = rv[:, None, None]
            integrand = ((1.0 - cdf) * (q_raw - y_b) if upper else cdf * (y_b - q_raw)) * phi
            return np.sum(wts * integrand, axis=(1, 2))

        lower = _arm(a, z_split, upper=False)
        upper_arm = _arm(z_split, b, upper=True)
        return np.asarray(2.0 * (lower + upper_arm), dtype=float)


# ---------------------------------------------------------------------------
# Stage 1: seeded deterministic mean baseline
# ---------------------------------------------------------------------------


class RidgeMeanForecaster:
    """Stage-1 deterministic mean forecaster: seeded closed-form ridge.

    arXiv:2608.11114 §4 admits ANY point predictor trained under a
    mean-focused loss as Stage 1 ("TORF is model-agnostic at this stage"; the
    paper uses SimpleTM). The repo's theta/dlinear estimators consume series
    windows and do not import into the ``(X, y)`` harness contract, so this
    inline ridge baseline plays the deterministic mean role: center features
    and target, solve the penalized normal equations
    ``(Z'Z + alpha I) w = Z'(y - ybar)`` in closed form. No iterative RNG:
    bitwise deterministic given identical inputs.
    """

    def __init__(self, alpha: float = 1e-2) -> None:
        if isinstance(alpha, bool) or not math.isfinite(float(alpha)) or float(alpha) < 0.0:
            raise ValueError(f"alpha must be finite and >= 0; got {alpha!r}")
        self.alpha = float(alpha)
        self._coef: Array | None = None
        self._x_mean: Array | None = None
        self._y_mean = 0.0
        self._n_features = 0

    @property
    def is_fitted(self) -> bool:
        return self._coef is not None

    def fit(self, X: Array, y: Array) -> RidgeMeanForecaster:
        Xm = _check_matrix(X, "X")
        yv = _check_vector(y, Xm.shape[0], "y")
        if Xm.shape[0] < Xm.shape[1] + 2:
            raise ValueError(
                f"too few samples for a ridge fit: need >= p + 2 = {Xm.shape[1] + 2}, "
                f"got {Xm.shape[0]}"
            )
        self._x_mean = np.asarray(Xm.mean(axis=0), dtype=float)
        self._y_mean = float(yv.mean())
        Z = Xm - self._x_mean
        yc = yv - self._y_mean
        G = Z.T @ Z
        G[np.diag_indices_from(G)] += self.alpha
        try:
            coef = np.linalg.solve(G, Z.T @ yc)
        except np.linalg.LinAlgError as exc:
            raise ValueError("ridge normal equations are singular; increase alpha") from exc
        if not bool(np.all(np.isfinite(coef))):
            raise ValueError("ridge solution is not finite; increase alpha")
        self._coef = np.asarray(coef, dtype=float)
        self._n_features = int(Xm.shape[1])
        return self

    def predict(self, X: Array) -> Array:
        if self._coef is None or self._x_mean is None:
            raise RuntimeError("RidgeMeanForecaster is not fitted")
        Xm = _check_matrix(X, "X")
        if Xm.shape[1] != self._n_features:
            raise ValueError(
                f"feature count mismatch: fitted with {self._n_features}, got {Xm.shape[1]}"
            )
        return np.asarray(self._y_mean + (Xm - self._x_mean) @ self._coef, dtype=float)


# ---------------------------------------------------------------------------
# The two-stage model
# ---------------------------------------------------------------------------


class TORFForecaster:
    """Two-stage Odd Residual Flows forecaster (arXiv:2608.11114).

    Stage 1 fits a deterministic mean; Stage 2 fits an odd residual flow
    around the FROZEN Stage-1 output. :meth:`predict` returns the Stage-1
    forecast verbatim: the predictive mean AND median equal the point
    forecast exactly (Lemma 1), so density estimation can never degrade point
    accuracy -- the structural advantage over joint NLL training (Seitzer et
    al. 2022) and over sampling-based flows/diffusions. Any ``stage1`` object
    exposing ``fit(X, y)`` / ``predict(X)`` can be injected.

    Proper scores only (CRPS, log score, PIT) plus point MAE; SYNTHETIC
    correctness material, never market evidence.
    """

    def __init__(self, stage1: Any = None, flow: OddResidualFlow | None = None) -> None:
        s1 = stage1 if stage1 is not None else RidgeMeanForecaster()
        if not (callable(getattr(s1, "fit", None)) and callable(getattr(s1, "predict", None))):
            raise ValueError("stage1 must expose fit(X, y) and predict(X)")
        fl = flow if flow is not None else OddResidualFlow()
        if not isinstance(fl, OddResidualFlow):
            raise ValueError("flow must be an OddResidualFlow")
        self.stage1 = s1
        self.flow = fl
        self._n_features = 0

    @property
    def is_fitted(self) -> bool:
        return bool(self.flow.is_fitted)

    def _check_xy(self, X: Array, y: Array | None) -> tuple[Array, Array | None]:
        Xm = _check_matrix(X, "X")
        if y is None:
            return Xm, None
        return Xm, _check_vector(y, Xm.shape[0], "y")

    def _stage1_mu(self, X: Array) -> Array:
        mu = np.asarray(self.stage1.predict(X), dtype=float).ravel()
        if mu.shape[0] != X.shape[0] or not bool(np.all(np.isfinite(mu))):
            raise ValueError("stage-1 prediction must be finite and match the row count of X")
        return mu

    def _require_fitted(self) -> None:
        if not self.flow.is_fitted:
            raise RuntimeError("TORFForecaster is not fitted")

    def fit(self, X: Array, y: Array) -> TORFForecaster:
        Xm, y_ = self._check_xy(X, y)
        if not (y_ is not None):
            raise ValueError("y_ is not None")
        yv = y_
        self.stage1.fit(Xm, yv)
        mu = self._stage1_mu(Xm)
        self.flow.fit(Xm, yv - mu)
        self._n_features = int(Xm.shape[1])
        return self

    def predict(self, X: Array) -> Array:
        """Point forecast: EXACTLY the frozen Stage-1 mean (never refit/resampled)."""
        self._require_fitted()
        Xm, _ = self._check_xy(X, None)
        if Xm.shape[1] != self._n_features:
            raise ValueError(
                f"feature count mismatch: fitted with {self._n_features}, got {Xm.shape[1]}"
            )
        return self._stage1_mu(Xm)

    def _residual(self, X: Array, y: Array) -> tuple[Array, Array]:
        mu = self.predict(X)
        yv = _check_vector(y, X.shape[0], "y")
        return mu, np.asarray(yv - mu, dtype=float)

    def predict_quantiles(self, X: Array, taus: Array) -> Array:
        """Predictive quantiles ``mu + Q_flow(tau)``, shape (n, T), non-crossing."""
        mu = self.predict(X)
        q = self.flow.quantile(X, taus)
        return np.asarray(mu[:, None] + q, dtype=float)

    def cdf(self, X: Array, y: Array) -> Array:
        """Analytic predictive CDF at ``y``: ``F(y - mu(X))`` (no sampling)."""
        _, r = self._residual(X, y)
        return self.flow.cdf(X, r)

    def pit(self, X: Array, y: Array) -> Array:
        """PIT values ``F(y | x)``; uniform(0, 1) under a calibrated fit."""
        return self.cdf(X, y)

    def crps(self, X: Array, y: Array) -> float:
        """Mean CRPS of the predictive distribution (proper score, lower better)."""
        _, r = self._residual(X, y)
        return float(np.mean(self.flow.crps(X, r)))

    def log_score(self, X: Array, y: Array) -> float:
        """Mean predictive log density at ``y`` (proper score, higher better)."""
        _, r = self._residual(X, y)
        return float(np.mean(self.flow.log_prob(X, r)))

    def sample(self, X: Array, n_samples: int, seed: int = 0) -> Array:
        """Seeded predictive draws ``mu + eps``, shape (n, M) (Appendix D inverse pass)."""
        mu = self.predict(X)
        eps = self.flow.sample(X, n_samples, seed=seed)
        return np.asarray(mu[:, None] + eps, dtype=float)

    def mae(self, X: Array, y: Array) -> float:
        """Point MAE of the frozen Stage-1 mean -- TORF's point accuracy."""
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

    Adapts the paper's Appendix C DGP (deterministic mean plus input-dependent
    symmetric Student-t noise, so ``E[y | x] = mu(x)`` exactly and mean
    preservation is testable) to the harness ``(X, y)`` tabular contract. The
    mean ``mu = 2*x0 + x1^2`` is linear in the provided features, and the
    scale ``gamma(x) = 0.3 + 1.2*(x1 + 1)/2`` is heteroskedastic.
    ``noise='student_t'``: symmetric heavy tails (nu = 4, variance-normalized
    by gamma); ``noise='bimodal'``: symmetric two-point residual of width
    gamma with slight jitter. Both violate the Gaussian residual assumption
    of MVE/NGBoost-style baselines while keeping the conditional mean exact.
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


def bench_odd_residual_flows(
    n_train: int = 1000,
    n_test: int = 600,
    seed: int = 0,
    noise: str = "student_t",
    n_blocks: int = 2,
    n_bins: int = 8,
    epochs: int = 250,
    lr: float = 8e-3,
    ngboost_rounds: int = 80,
    ngboost_lr: float = 0.1,
) -> dict[str, float | str]:
    """TORF vs NGBoostGaussian on a SYNTHETIC seeded heteroskedastic stream.

    Correctness bench for the paper's central claim (arXiv:2608.11114 §5):
    the odd residual flow matches or beats the Gaussian density baseline on
    CRPS while preserving the Stage-1 point forecast EXACTLY --
    ``mae_preservation_gap`` is 0.0 by construction, because TORF's analytic
    mean IS the frozen Stage-1 output (Lemma 1). ``NGBoostGaussian`` (Duan et
    al. 2020; ``quant_fund.models.ngboost_lite``) is fit with
    ``score='crps'`` -- the same proper score used for the comparison. Both
    models see identical features and data; everything is seeded. Proper
    scores plus point MAE only; labeled ``dgp=fixture`` /
    ``claim=research_metric_only``; never market evidence, no Sharpe/P&L
    (AGENTS.md honesty contract). Requires the torch ``nn`` extra for the
    Stage-2 fit.
    """
    from quant_fund.metrics.probability import pit_ks
    from quant_fund.models.ngboost_lite import NGBoostGaussian

    Xtr, ytr, Xte, yte = _synthetic_hetero_stream(n_train, n_test, seed, noise)
    torf = TORFForecaster(
        flow=OddResidualFlow(n_blocks=n_blocks, n_bins=n_bins, epochs=epochs, lr=lr, seed=int(seed))
    ).fit(Xtr, ytr)
    ngb = NGBoostGaussian(
        n_estimators=ngboost_rounds, learning_rate=ngboost_lr, score="crps", seed=int(seed)
    ).fit(Xtr, ytr)

    mu = torf.predict(Xte)
    stage1_mu = np.asarray(torf.stage1.predict(Xte), dtype=float)
    torf_mae = float(np.mean(np.abs(yte - mu)))
    stage1_mae = float(np.mean(np.abs(yte - stage1_mu)))
    torf_crps = torf.crps(Xte, yte)
    ngb_crps = float(ngb.crps(Xte, yte))
    samples = torf.sample(Xte, 2000, seed=int(seed) + 1)
    q = torf.predict_quantiles(Xte, np.array([0.05, 0.95]))
    pit_stat, pit_p = pit_ks(torf.pit(Xte, yte))
    return {
        "synthetic_torf_crps": torf_crps,
        "synthetic_ngboost_crps": ngb_crps,
        "synthetic_crps_gain_vs_ngboost": ngb_crps - torf_crps,
        "synthetic_torf_log_score": torf.log_score(Xte, yte),
        "synthetic_ngboost_log_score": float(ngb.log_score(Xte, yte)),
        "synthetic_torf_mae": torf_mae,
        "synthetic_stage1_mae": stage1_mae,
        "synthetic_mae_preservation_gap": abs(torf_mae - stage1_mae),
        "synthetic_residual_sample_mean": float(np.mean(samples - mu[:, None])),
        "synthetic_coverage_90": float(np.mean((yte >= q[:, 0]) & (yte <= q[:, 1]))),
        "synthetic_pit_ks": float(pit_stat),
        "synthetic_pit_ks_pvalue": float(pit_p),
        "synthetic_n_train": float(n_train),
        "synthetic_n_test": float(n_test),
        "synthetic_seed": float(seed),
        "synthetic_noise": str(noise),
        "synthetic_dgp": "fixture",
        "synthetic_claim": "research_metric_only",
        "synthetic_synthetic": "heteroskedastic_symmetric_seeded",
    }
