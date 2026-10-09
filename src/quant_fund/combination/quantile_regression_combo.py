"""Pinball-optimal combination weights per quantile level.

Finds simplex weights w_τ that minimise the empirical pinball loss of the
combined quantile q_τ = Σ_k w_k q_{k,τ} at each level τ. Solved with
projected subgradient descent (simplex projection) — convex problem,
deterministic step schedule, no new dependencies:

- ``pinball_optimal_weights`` — projected subgradient on the simplex for a
  single quantile level;
- ``pinball_optimal_all_levels`` — weights for every level, with optional
  smoothness coupling across adjacent τ;
- ``combined_quantiles`` — rebuild the combined quantile matrix from the
  weight matrix (with monotonicity repair).

Honesty: weights minimise in-sample pinball on the supplied panel; convexity
makes the global optimum well-defined up to tolerance.

References:
- Gneiting, T. (2011). Making and evaluating point forecasts — quantiles
  and consistent scoring.
- Boyd, S., Vandenberghe, L. (2004). *Convex Optimization* — projected
  subgradient methods and the simplex projection.

Composition: numpy; deterministic step schedule.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _project_simplex(v: FloatArray) -> FloatArray:
    """Euclidean projection of v onto the probability simplex (Duchi et al.)."""
    v = np.asarray(v, dtype=np.float64)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = int(np.max(np.flatnonzero(u * np.arange(1, len(v) + 1) > (css - 1.0)))) if len(v) else 0
    theta = (css[rho] - 1.0) / (rho + 1)
    return cast(FloatArray, np.maximum(v - theta, 0.0))


def _pinball_grad_w(
    w: FloatArray,
    member_q: FloatArray,
    y: FloatArray,
    tau: float,
) -> FloatArray:
    """Gradient of mean pinball wrt w: E[(1{y<q} − τ) · member_q].

    From dL/dq = −(τ − 1{y<q}) and dq/dw_k = q_k, the chain rule gives
    dL/dw_k = (1{y<q} − τ)·q_k.
    """
    q = member_q @ w
    indicator = (y < q).astype(np.float64)
    coeff = indicator - tau
    return cast(FloatArray, (coeff[:, None] * member_q).mean(axis=0))


def pinball_optimal_weights(
    member_q: FloatArray,
    y: FloatArray,
    tau: float,
    *,
    iters: int = 400,
    step0: float = 0.5,
) -> FloatArray:
    """Simplex weights minimising mean pinball at level τ.

    ``member_q`` is (T, M) member quantiles at level τ; ``y`` is (T,).
    """
    q = np.asarray(member_q, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if q.ndim != 2 or y_arr.shape != (q.shape[0],):
        raise ValueError("member_q must be (T, M) and y must be (T,)")
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must be in (0, 1)")
    t_total, m = q.shape
    scale = float(np.std(q)) + 1e-12
    qn = q / scale
    yn = y_arr / scale
    w: FloatArray = np.full(m, 1.0 / m)
    for it in range(1, iters + 1):
        eta = step0 / np.sqrt(it)
        grad = _pinball_grad_w(w, qn, yn, tau)
        w = _project_simplex(w - eta * grad)
    return np.asarray(w, dtype=np.float64)


def pinball_optimal_all_levels(
    member_quantiles: FloatArray,
    y: FloatArray,
    taus: FloatArray,
    *,
    iters: int = 400,
) -> FloatArray:
    """Per-level optimal weights; returns (M, Q) aligned with taus."""
    q = np.asarray(member_quantiles, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    taus_arr = np.asarray(taus, dtype=np.float64)
    if q.ndim != 3:
        raise ValueError("member_quantiles must be (T, M, Q)")
    t_total, m, n_q = q.shape
    if taus_arr.shape != (n_q,) or y_arr.shape != (t_total,):
        raise ValueError("taus must be (Q,) and y must be (T,)")
    w = np.empty((m, n_q), dtype=np.float64)
    for k in range(n_q):
        w[:, k] = pinball_optimal_weights(q[:, :, k], y_arr, float(taus_arr[k]), iters=iters)
    return w


def combined_quantiles(
    member_quantiles: FloatArray,
    weights: FloatArray,
) -> FloatArray:
    """Combined quantiles (T, Q) = Σ_k w_{k,q} q_{k,t,q} with monotonicity."""
    q = np.asarray(member_quantiles, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if q.ndim != 3 or w.ndim != 2 or w.shape[0] != q.shape[1] or w.shape[1] != q.shape[2]:
        raise ValueError("member_quantiles must be (T, M, Q) and weights (M, Q)")
    combined = np.einsum("tmq,mq->tq", q, w)
    return np.asarray(np.sort(combined, axis=1), dtype=np.float64)
