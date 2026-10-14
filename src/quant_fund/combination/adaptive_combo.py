"""Regime-feature-adaptive combination weights.

Learns combination weights as a function of observable features (e.g. a
volatility or trend indicator) so the ensemble can adapt when the identity
of the best member changes with the regime:

- ``fit_adaptive_weights`` — ridge regression from features to per-member
  pinball-responsibility logits on a training window; the fitted map is
  w_t = softmax(X_t β), so weights are a smooth function of the regime;
- ``adaptive_combine`` — apply the fitted map out of sample;
- ``adaptive_vs_static`` — out-of-sample mean pinball of the adaptive
  ensemble versus the static equal-weight ensemble.

Honesty: the feature→weight map is fitted on the training window supplied;
gains are measured out of sample on the supplied evaluation stream.

References:
- Elliott, G., Timmermann, A. (2005). Optimal forecast combination under
  regime switching — adaptive combination.
- Bates, J. M., Granger, C. W. J. (1969). The combination of forecasts.

Composition: numpy + scipy.optimize (locked); deterministic seeds.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _softmax(z: FloatArray) -> FloatArray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return cast(FloatArray, e / e.sum(axis=1, keepdims=True))


def fit_adaptive_weights(
    features: FloatArray,
    member_quantiles: FloatArray,
    y: FloatArray,
    taus: FloatArray,
    *,
    ridge: float = 1e-3,
) -> dict[str, FloatArray]:
    """Fit w_t = softmax(X_t β) by ridge regression on responsibility logits.

    ``features`` is (T, D); ``member_quantiles`` is (T, M, Q) aligned with
    ``taus`` (Q,); ``y`` is (T,). Responsibilities are built from negative
    mean pinball per member (higher is better), then β is the ridge solution
    mapping features to responsibility logits. The fit minimises squared
    error on logits — a tractable surrogate for the softmax regression.
    """
    x = np.asarray(features, dtype=np.float64)
    q = np.asarray(member_quantiles, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    t_total = x.shape[0]
    if q.ndim != 3 or q.shape[0] != t_total:
        raise ValueError("member_quantiles must be (T, M, Q)")
    n_q = q.shape[2]
    taus_arr = np.asarray(taus, dtype=np.float64)
    if taus_arr.shape != (n_q,):
        raise ValueError("taus must be (Q,)")
    if y_arr.shape != (t_total,):
        raise ValueError("y must be (T,)")
    d = x.shape[1]
    # responsibilities: negative mean pinball per member (T, M)
    diff = y_arr[:, None, None] - q
    loss = np.maximum(taus_arr[None, None, :] * diff, (taus_arr[None, None, :] - 1.0) * diff)
    resp = -np.mean(loss, axis=2)
    resp = resp - resp.mean(axis=1, keepdims=True)
    # ridge from [1, X] to resp
    design = np.column_stack([np.ones(t_total), x])
    gram = design.T @ design
    reg = gram + ridge * np.eye(d + 1)
    beta = np.linalg.solve(reg, design.T @ resp)
    return {"beta": np.asarray(beta, dtype=np.float64)}


def adaptive_combine(
    features: FloatArray,
    member_quantiles: FloatArray,
    beta: FloatArray,
) -> dict[str, FloatArray]:
    """Apply the fitted map: w_t = softmax([1, X_t] β); combined quantiles."""
    x = np.asarray(features, dtype=np.float64)
    q = np.asarray(member_quantiles, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    if q.ndim != 3 or q.shape[0] != x.shape[0]:
        raise ValueError("member_quantiles must be (T, M, Q) matching features")
    if b.shape != (x.shape[1] + 1, q.shape[1]):
        raise ValueError("beta has incompatible shape with features/members")
    design = np.column_stack([np.ones(x.shape[0]), x])
    logits = design @ b
    w = _softmax(logits)
    combined = np.einsum("tm,tmq->tq", w, q)
    return {
        "weights": np.asarray(w, dtype=np.float64),
        "combined": np.asarray(combined, dtype=np.float64),
    }


def adaptive_vs_static(
    features: FloatArray,
    member_quantiles: FloatArray,
    y: FloatArray,
    taus: FloatArray,
    *,
    train_frac: float = 0.5,
    ridge: float = 1e-3,
) -> dict[str, float]:
    """Out-of-sample mean pinball: adaptive ensemble vs equal weights.

    Train on the first ``train_frac`` of periods and evaluate on the rest.
    """
    x = np.asarray(features, dtype=np.float64)
    q = np.asarray(member_quantiles, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    taus_arr = np.asarray(taus, dtype=np.float64)
    t_total = x.shape[0]
    t_cut = int(t_total * train_frac)
    if t_cut < 20 or t_total - t_cut < 20:
        raise ValueError("need at least 20 periods in train and test")
    fit = fit_adaptive_weights(x[:t_cut], q[:t_cut], y_arr[:t_cut], taus_arr, ridge=ridge)
    out = adaptive_combine(x[t_cut:], q[t_cut:], fit["beta"])
    combined = out["combined"]
    diff = y_arr[t_cut:, None] - combined
    loss_adaptive = float(np.mean(np.maximum(taus_arr * diff, (taus_arr - 1.0) * diff)))
    static_q = np.mean(q[t_cut:], axis=1)
    diff_s = y_arr[t_cut:, None] - static_q
    loss_static = float(np.mean(np.maximum(taus_arr * diff_s, (taus_arr - 1.0) * diff_s)))
    return {
        "adaptive_pinball": loss_adaptive,
        "static_pinball": loss_static,
        "train_periods": float(t_cut),
        "test_periods": float(t_total - t_cut),
    }
