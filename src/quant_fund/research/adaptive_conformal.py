"""Adaptive conformal inference under distribution drift (ACI).

Gibbs & Candes (2021): split conformal's fixed miscoverage alpha breaks
under shift — coverage drifts away from 1-alpha as the residual scale
moves. ACI treats alpha as an online quantile-tracking problem:

    alpha_{t+1} = clip(alpha_t + step * (alpha_target - err_t), lo, hi)

where err_t = 1{y_t outside the (alpha_t)-level interval}. The update is
a Robbins–Monro iteration on the breach indicator: at stationarity
alpha_t oscillates around the alpha that delivers the target rate, so
long-run empirical coverage approaches the target regardless of drift.
The conformal set at time t uses quantile level (1 - alpha_t) on the
rolling calibration residuals.

- ``AdaptiveConformal`` — online interval: feeds residuals |y - qhat|,
  emits the next interval half-width and tracks alpha_t.
- ``aci_run`` — apply to a (residual, realized) stream, return alpha
  path + per-step coverage + half-widths.
- ``adaptive_conformal_bench`` — sealed ``adaptive_conformal.v1``:
  static split-conformal vs ACI on a drifting synthetic stream (variance
  step at mid-stream): static interval over/under-covers around the
  shift while ACI re-converges; metrics are interval coverage MAE vs the
  target and post-shift recovery time.

SYNTHETIC only.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_revision

ACI_SCHEMA = "adaptive_conformal.v1"


def _prob(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v <= 0.0 or v >= 1.0:
        raise ValueError(f"{name} must be in (0,1), got {x!r}")
    return v


def _pos(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {x!r}")
    return v


class AdaptiveConformal:
    """ACI interval tracker over a residual stream.

    ``update(score_t)`` returns the half-width for the *next* point and
    records coverage; the score is the conformity residual of the point
    just realized (|y - center| for symmetric intervals).
    """

    def __init__(
        self,
        *,
        alpha: float,
        step: float = 0.05,
        calib_size: int = 200,
        alpha_bounds: tuple[float, float] = (0.001, 0.5),
    ) -> None:
        self.alpha_target = _prob(alpha, "alpha")
        self.step = _pos(step, "step")
        if calib_size < 10:
            raise ValueError(f"calib_size must be >= 10, got {calib_size!r}")
        self.calib_size = int(calib_size)
        lo, hi = alpha_bounds
        _prob(lo, "alpha_bounds[0]")
        _prob(hi, "alpha_bounds[1]")
        if not lo < hi:
            raise ValueError("alpha_bounds must satisfy lo < hi")
        self.lo, self.hi = float(lo), float(hi)
        self._alpha = self.alpha_target
        self._scores: list[float] = []

    @property
    def alpha_t(self) -> float:
        return self._alpha

    @property
    def n(self) -> int:
        return len(self._scores)

    def update(self, score: float) -> float:
        """Record the realized conformity score; return next half-width."""
        if not math.isfinite(score) or score < 0.0:
            raise ValueError(f"score must be non-negative and finite, got {score!r}")
        window = self._scores[-self.calib_size :]
        half = _rolling_half_width(window, self._alpha)
        err = 1.0 if score > half else 0.0
        self._alpha = float(
            np.clip(self._alpha + self.step * (self.alpha_target - err), self.lo, self.hi)
        )
        self._scores.append(score)
        return _rolling_half_width(self._scores[-self.calib_size :], self._alpha)


def _rolling_half_width(window: list[float], alpha: float) -> float:
    if not window:
        return float("inf")
    return float(np.quantile(np.asarray(window), 1.0 - alpha))


def static_conformal_width(
    scores: NDArray[np.float64], alpha: float, *, calib_size: int
) -> NDArray[np.float64]:
    """Rolling static split-conformal half-widths (fixed alpha)."""
    _prob(alpha, "alpha")
    if calib_size < 10:
        raise ValueError(f"calib_size must be >= 10, got {calib_size!r}")
    n = len(scores)
    out = np.full(n, np.inf)
    for t in range(1, n):
        out[t] = _rolling_half_width(list(scores[max(0, t - calib_size) : t]), alpha)
    return out


def aci_run(
    scores: NDArray[np.float64],
    *,
    alpha: float,
    step: float = 0.05,
    calib_size: int = 200,
    alpha_bounds: tuple[float, float] = (0.001, 0.5),
) -> dict[str, Any]:
    """Run ACI over a score stream; return alpha/half-width/coverage paths."""
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 1 or x.size < 20:
        raise ValueError("scores must be a 1-D array with >= 20 points")
    if np.any(~np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("scores must be non-negative and finite")
    aci = AdaptiveConformal(
        alpha=alpha, step=step, calib_size=calib_size, alpha_bounds=alpha_bounds
    )
    alphas = np.empty(x.size)
    halves = np.empty(x.size)
    covered = np.zeros(x.size, dtype=np.float64)
    # first half-width is +inf (no calibration yet) — count as covered
    prev_half = float("inf")
    for t, s in enumerate(x):
        alphas[t] = aci.alpha_t
        halves[t] = prev_half
        covered[t] = float(s <= prev_half)
        prev_half = aci.update(float(s))
    halves[0] = float("nan")  # +inf sentinel reads better as nan
    return {
        "alphas": alphas,
        "half_widths": halves,
        "covered": covered,
    }


def _drifting_scores(n: int, *, seed: int, shift_at: int | None = None) -> NDArray[np.float64]:
    """Half-normal residual stream with a variance step at ``shift_at``."""
    if n < 40:
        raise ValueError("n must be >= 40")
    rng = np.random.default_rng(seed)
    shift_at = n // 2 if shift_at is None else shift_at
    scale = np.ones(n)
    scale[shift_at:] = 3.0
    return np.abs(rng.normal(0.0, 1.0, n)) * scale


def _recovery_time(
    covered: NDArray[np.float64],
    shift_at: int,
    target: float,
    *,
    tol: float,
    window: int = 25,
) -> int:
    """First index >= shift_at whose trailing-window coverage hits tol band."""
    n = len(covered)
    for t in range(shift_at, n - window):
        if abs(np.mean(covered[t : t + window]) - target) <= tol:
            return t - shift_at
    return n - shift_at


def adaptive_conformal_bench(
    *,
    n: int = 1200,
    alpha: float = 0.1,
    step: float = 0.05,
    calib_size: int = 150,
    seed: int = 0,
) -> dict[str, Any]:
    """Sealed ``adaptive_conformal.v1`` receipt.

    Static rolling split-conformal vs ACI on a variance-shift stream.
    Claims: coverage MAE vs the target before/after the shift and ACI's
    post-shift recovery time (smaller is better); all proper/coverage
    metrics, never P&L.
    """
    if n < 200:
        raise ValueError(f"n must be >= 200, got {n!r}")
    alpha = _prob(alpha, "alpha")
    step = _pos(step, "step")
    if calib_size < 10 or calib_size >= n // 2:
        raise ValueError(f"calib_size must be in [10, n/2), got {calib_size!r}")
    scores = _drifting_scores(n, seed=seed)
    shift_at = n // 2

    stat_w = static_conformal_width(scores, alpha, calib_size=calib_size)
    stat_cov = (scores <= stat_w).astype(np.float64)
    stat_cov[0] = 1.0

    out = aci_run(
        scores,
        alpha=alpha,
        step=step,
        calib_size=calib_size,
        alpha_bounds=(0.001, 0.5),
    )
    aci_cov = out["covered"]

    burn = calib_size + 10
    pre = slice(burn, shift_at)
    post = slice(shift_at + burn // 4, n)

    cov_target = 1.0 - alpha

    def cov_mae(c: NDArray[np.float64], s: slice) -> float:
        return float(abs(np.mean(c[s]) - cov_target))

    payload: dict[str, Any] = {
        "schema": ACI_SCHEMA,
        "kind": "adaptive_conformal",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "disclaimer": (
            "ACI vs static split-conformal on a synthetic drifting "
            "residual stream; never market evidence."
        ),
        "n": n,
        "alpha_target": alpha,
        "step": step,
        "calib_size": calib_size,
        "shift_at": shift_at,
        "static_coverage_pre": float(np.mean(stat_cov[pre])),
        "static_coverage_post": float(np.mean(stat_cov[post])),
        "aci_coverage_pre": float(np.mean(aci_cov[pre])),
        "aci_coverage_post": float(np.mean(aci_cov[post])),
        "static_mae_pre": cov_mae(stat_cov, pre),
        "static_mae_post": cov_mae(stat_cov, post),
        "aci_mae_pre": cov_mae(aci_cov, pre),
        "aci_mae_post": cov_mae(aci_cov, post),
        "aci_recovery_time_events": _recovery_time(aci_cov, shift_at, cov_target, tol=alpha),
        "static_recovery_time_events": _recovery_time(stat_cov, shift_at, cov_target, tol=alpha),
        "alpha_path_final": float(out["alphas"][-1]),
        "alpha_path_min": float(np.min(out["alphas"])),
        "alpha_path_max": float(np.max(out["alphas"])),
    }
    payload["payload_sha256"] = hash_bytes(json.dumps(payload, sort_keys=True).encode())
    return payload


__all__ = [
    "ACI_SCHEMA",
    "AdaptiveConformal",
    "aci_run",
    "adaptive_conformal_bench",
    "static_conformal_width",
]
