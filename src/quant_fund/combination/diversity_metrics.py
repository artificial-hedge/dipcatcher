"""Ensemble diversity metrics via the ambiguity decomposition.

For convex-weighted combinations the expected loss decomposes exactly:

    L(ŝ) = Σ w_k L(s_k) − Σ w_k E[(s_k − ŝ)²]  + ... (for squared loss)

so diversity (spread between members) *reduces* the ensemble loss for a
given member quality. This module exposes:

- ``ambiguity_decomposition`` — the exact split for squared loss:
  ensemble MSE = weighted member MSE − weighted variance of member
  outputs;
- ``diversity_bonus`` — how much of the ensemble advantage over the
  weighted member loss comes from diversity;
- ``effective_ensemble_size`` — inverse of the herfindahl of the weights
  (1/Σ w²) — the count of effectively independent members.

Honesty: the decomposition is exact for squared loss by construction; the
"diversity bonus" is descriptive on the supplied window.

References:
- Hansen, J. V., Salamon, P. (1990). Neural network ensembles — the
  ambiguity decomposition.
- Krogh, A., Vedelsby, J. (1995). Neural network ensembles, cross
  validation, and active learning — the L = L̄ − Ā form used here.
- López de Prado, M. (2020). *Machine Learning for Asset Managers* —
  effective ensemble size.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def ambiguity_decomposition(
    member_preds: FloatArray,
    y: FloatArray,
    weights: FloatArray,
) -> dict[str, float]:
    """Squared-loss decomposition: L_comb = L̄_members − Ā_diversity.

    ``member_preds`` is (T, M), ``y`` is (T,), ``weights`` is (M,) and
    normalised internally. Returns the ensemble MSE, weighted member MSE,
    the diversity term, and the residual identity error (should be ~0).
    """
    f = np.asarray(member_preds, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if f.ndim != 2 or y_arr.shape != (f.shape[0],) or w.shape != (f.shape[1],):
        raise ValueError("member_preds (T, M), y (T,), weights (M,) must align")
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total <= 0:
        w = np.full(len(w), 1.0 / len(w))
    else:
        w = w / total
    comb = f @ w
    member_mse = float(w @ np.mean((f - y_arr[:, None]) ** 2, axis=0))
    diversity = float(w @ np.mean((f - comb[:, None]) ** 2, axis=0))
    comb_mse = float(np.mean((comb - y_arr) ** 2))
    residual = comb_mse - (member_mse - diversity)
    return {
        "ensemble_mse": comb_mse,
        "member_mse": member_mse,
        "diversity_term": diversity,
        "residual": residual,
    }


def diversity_bonus(member_preds: FloatArray, y: FloatArray, weights: FloatArray) -> float:
    """Fraction of the member-weighted MSE removed by diversity, in [0, 1]."""
    out = ambiguity_decomposition(member_preds, y, weights)
    if out["member_mse"] <= 0:
        return 0.0
    return float(np.clip(out["diversity_term"] / out["member_mse"], 0.0, 1.0))


def effective_ensemble_size(weights: FloatArray) -> float:
    """1 / Σ w² — the number of effectively independent members."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or len(w) == 0:
        raise ValueError("weights must be a non-empty one-dimensional array")
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total <= 0:
        return float(len(w))
    w = w / total
    return float(1.0 / np.sum(w * w))
