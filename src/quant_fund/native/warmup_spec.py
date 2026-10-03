"""Warmup specification for chunked/streaming kernel execution.

Sharded replay applies a kernel to ``series[seam-k:]`` and trusts outputs
from ``seam`` onward. That is only sound when the kernel's state at ``seam``
depends on at most ``k`` earlier inputs — and even then, floating-point
reassociation (cumsum-difference rolling sums, recursive EMAs) means chunked
output usually converges to, rather than reproduces, the full-series result.

``warmup_spec`` measures the seam residual as a function of warmup and
classifies each kernel:

- ``exact``: residual reaches literal 0 (stateless-per-window pair ops like
  ``simple_returns``, which need exactly ``w - 1`` warmup bars of inputs).
- ``floored``: residual stalls at float noise (<= ``floor_tol``) after a
  finite warmup — chunk-safe in practice (rolling cumsum ops land here).
- ``decaying``: residual strictly shrinks with warmup but stays above the
  floor inside ``max_warmup`` (recursive kernels; warmup buys precision
  geometrically).
- ``nonlocal``: residual does not decay — absolute state survives any
  prefix drop (``wealth_index`` restarts at 1 wherever the chunk begins).

The table is the contract a streaming deployment must honor.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quant_fund.native import reference
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

FloatArray = NDArray[np.float64]
Op = Callable[[FloatArray], FloatArray]

FLOOR_TOL = 1e-8
_GRID = (0, 1, 2, 4, 8, 16, 32, 64, 128)


def _seam_max_diff(full: FloatArray, chunk_tail: FloatArray) -> float:
    """Max |a-b| at the seam onward; NaN==NaN counts equal, inf must match."""
    if full.shape != chunk_tail.shape:
        raise ValueError(f"shape mismatch {full.shape} vs {chunk_tail.shape}")
    both_nan = np.isnan(full) & np.isnan(chunk_tail)
    same_inf = (full == chunk_tail) & ~np.isfinite(full)
    with np.errstate(invalid="ignore"):
        diff = np.abs(full - chunk_tail)
    diff = np.where(both_nan | same_inf, 0.0, diff)
    diff = np.where(np.isnan(diff), np.inf, diff)
    return float(diff.max()) if diff.size else 0.0


def seam_residual(fn: Op, series: ArrayLike, seam: int, warmup: int) -> float:
    """Max deviation a ``warmup``-bar prefix leaves at indices >= ``seam``."""
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("warmup_spec kernels are 1-d")
    n = arr.size
    if not 0 < seam < n:
        raise ValueError(f"seam {seam} outside (0, {n})")
    if not 0 <= warmup <= seam:
        raise ValueError(f"warmup {warmup} outside [0, {seam}]")
    full = np.asarray(fn(arr), dtype=np.float64)
    chunk = np.asarray(fn(arr[seam - warmup :]), dtype=np.float64)
    return _seam_max_diff(full[seam:], chunk[warmup:])


def residual_curve(
    fn: Op,
    series: ArrayLike,
    seam: int,
    *,
    warmup_grid: Sequence[int] = _GRID,
) -> list[dict[str, float]]:
    """(warmup, residual) points, increasing warmup up to ``seam``."""
    arr = np.asarray(series, dtype=np.float64)
    pts = []
    for k in warmup_grid:
        if k >= seam:
            break
        pts.append({"warmup": int(k), "residual": seam_residual(fn, arr, seam, int(k))})
    return pts


def measure_warmup(
    fn: Op,
    series: ArrayLike,
    seam: int,
    *,
    tol: float = 0.0,
    max_warmup: int = 128,
) -> int | None:
    """Smallest warmup with seam residual <= ``tol``; ``None`` if above cap."""
    for k in range(min(max_warmup, seam) + 1):
        if seam_residual(fn, series, seam, k) <= tol:
            return k
    return None


def _classify(
    fn: Op,
    series: ArrayLike,
    seam: int,
    curve: list[dict[str, float]],
    max_warmup: int,
) -> dict[str, Any]:
    # warmup == seam is the degenerate no-chunk point: never qualifies.
    bound = min(max_warmup, seam - 1)
    exact = measure_warmup(fn, series, seam, tol=0.0, max_warmup=bound)
    floored = measure_warmup(fn, series, seam, tol=FLOOR_TOL, max_warmup=bound)
    first, last = curve[0]["residual"], curve[-1]["residual"]
    if exact is not None:
        status = "exact"
    elif floored is not None:
        status = "floored"
    elif np.isfinite(last) and last < 0.1 * first:
        status = "decaying"
    else:
        status = "nonlocal"
    return {
        "status": status,
        "exact_warmup": exact,
        "warmup_at_floor": floored,
        "residual_first": first,
        "residual_last": last,
    }


def warmup_spec(
    ops: dict[str, Op],
    series: ArrayLike,
    seam: int,
    *,
    warmup_grid: Sequence[int] = _GRID,
) -> dict[str, dict[str, Any]]:
    """Per-op residual curve + warmup classification."""
    out: dict[str, dict[str, Any]] = {}
    for name, fn in ops.items():
        curve = residual_curve(fn, series, seam, warmup_grid=warmup_grid)
        entry = _classify(fn, series, seam, curve, max_warmup=seam - 1)
        entry["residual_curve"] = curve
        out[name] = entry
    return out


def _default_ops() -> dict[str, Op]:
    return {
        "rolling_mean_5": lambda s: reference.rolling_mean(s, 5),
        "rolling_mean_20": lambda s: reference.rolling_mean(s, 20),
        "rolling_std_5": lambda s: reference.rolling_std(s, 5),
        "rolling_std_20": lambda s: reference.rolling_std(s, 20),
        "ema_12": lambda s: reference.ema(s, 12),
        "ema_26": lambda s: reference.ema(s, 26),
        "rsi_14": lambda s: reference.rsi(s, 14),
        "bollinger_20_mid": lambda s: reference.bollinger(s, 20)["mid"],
        "bollinger_20_bandwidth": lambda s: reference.bollinger(s, 20)["bandwidth"],
        "simple_returns": reference.simple_returns,
        "wealth_index": lambda s: reference.wealth_index(reference.simple_returns(s)),
    }


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None  # non-finite residual: divergence is total
    if isinstance(value, dict):
        return {k: _finite_or_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(v) for v in value]
    return value


def warmup_spec_bench(*, n: int = 512, seam: int = 256, seed: int = 0) -> dict[str, Any]:
    """Sealed ``warmup_spec.v1`` receipt over the reference kernel suite.

    Non-finite residuals serialize as null — a divergent seam is a verdict,
    not a number, and must not break the canonical-JSON seal.
    """
    rng = np.random.default_rng(seed)
    series = 100.0 + np.cumsum(rng.standard_normal(n) * 0.4)
    table = _finite_or_none(warmup_spec(_default_ops(), series, seam))
    out: dict[str, Any] = {
        "kind": "warmup_spec",
        "schema": "warmup_spec.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "claim": {
            "invariant": "chunked replay at seam is safe iff warmup clears the residual curve",
            "n": n,
            "seam": seam,
            "floor_tol": FLOOR_TOL,
        },
        "interpretation": {
            "ops": table,
            "note": "nonlocal ops carry absolute state — shard only with a state handoff",
            "residual_null_means": "non-finite (inf/nan) divergence at that warmup",
        },
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


__all__ = [
    "FLOOR_TOL",
    "measure_warmup",
    "residual_curve",
    "seam_residual",
    "warmup_spec",
    "warmup_spec_bench",
]
