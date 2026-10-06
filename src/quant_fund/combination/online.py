"""Online forecast combination: exponentiated gradient and fixed-share.

Maintains combination weights as losses arrive:

- exponentiated gradient (EG): w_{t+1} ∝ w_t · exp(−η ℓ_t), the
  multiplicative-weights update with the classical regret bound
  ln K / η + η T L² / 8 for losses in [0, L];
- fixed-share (Herbster–Warmuth): mix a share α into the uniform
  distribution each step so weight can track a drifting best member.

Honesty: regret is measured against the best *fixed* member in hindsight on
the supplied loss sequence.

References:
- Littlestone, N., Warmuth, M. K. (1994). The weighted majority algorithm —
  multiplicative weights.
- Freund, Y., Schapire, R. E. (1997). A decision-theoretic generalization
  of on-line learning — exponentiated gradient.
- Herbster, M., Warmuth, M. K. (1998). Tracking the best expert — the
  fixed-share update.
- Cesa-Bianchi, N., Lugosi, G. (2006). *Prediction, Learning, and Games*.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _normalise(w: FloatArray) -> FloatArray:
    total = float(w.sum())
    if total <= 0 or not np.isfinite(total):
        return np.full(len(w), 1.0 / len(w))
    return w / total


def exponentiated_gradient(
    losses: FloatArray,
    eta: float,
) -> dict[str, FloatArray]:
    """EG combination on a (T, K) loss sequence (smaller is better)."""
    ell = np.asarray(losses, dtype=np.float64)
    if ell.ndim != 2:
        raise ValueError("losses must be (T, K)")
    if eta <= 0:
        raise ValueError("eta must be positive")
    t_total, k = ell.shape
    w = np.full(k, 1.0 / k)
    weight_path = np.empty((t_total, k), dtype=np.float64)
    combined = np.empty(t_total, dtype=np.float64)
    for t in range(t_total):
        weight_path[t] = w
        combined[t] = float(w @ ell[t])
        w = _normalise(w * np.exp(-eta * ell[t]))
    return {"weights": weight_path, "combined_loss": combined}


def fixed_share(
    losses: FloatArray,
    eta: float,
    alpha: float,
) -> dict[str, FloatArray]:
    """Fixed-share update: EG step then mix α into the uniform distribution."""
    ell = np.asarray(losses, dtype=np.float64)
    if ell.ndim != 2:
        raise ValueError("losses must be (T, K)")
    if eta <= 0:
        raise ValueError("eta must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    t_total, k = ell.shape
    w = np.full(k, 1.0 / k)
    weight_path = np.empty((t_total, k), dtype=np.float64)
    combined = np.empty(t_total, dtype=np.float64)
    for t in range(t_total):
        weight_path[t] = w
        combined[t] = float(w @ ell[t])
        w = _normalise(w * np.exp(-eta * ell[t]))
        w = (1.0 - alpha) * w + alpha * (1.0 / k)
    return {"weights": weight_path, "combined_loss": combined}


def regret_bound(t: int, k: int, eta: float, loss_range: float) -> float:
    """EG regret bound vs the best fixed member for losses in [0, L]:

    R_T ≤ ln K / η + η T L² / 8.
    """
    if t < 1 or k < 2:
        raise ValueError("need t >= 1 and k >= 2")
    if eta <= 0 or loss_range <= 0:
        raise ValueError("eta and loss_range must be positive")
    return float(np.log(k) / eta + eta * t * loss_range * loss_range / 8.0)
