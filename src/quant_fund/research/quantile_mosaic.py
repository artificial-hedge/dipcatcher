"""Quantile-mosaic consistency lane — a head's *internal* grid coherence.

The coverage/calibration lanes audit a head against realized outcomes. This
lane audits the quantile grid itself, plus streams PITs interpolated
*inside* the grid (the discrete-grid analogue of the probability integral
transform), which the outer breach-rate lanes cannot see:

- ``quantile_consistency`` — per-row checks: monotone non-decreasing
  quantiles (crossing), nonnegative interval widths, mid-vs-mean skew
  diagnostic. Strict mode raises on any crossing; audit mode counts it.
- ``QuantileMosaic`` — streams ``(quantiles_row, realized_y)`` and
  maintains the interpolated PIT estimate: ``y < q_min -> 0``,
  ``y > q_max -> 1``, else linear between the bracketing quantiles
  (piecewise-uniform CDF between grid knots — the standard convention).
- ``pit_histogram`` / per-tau breach rates.
- ``mosaic_bench`` — sealed ``quantile_mosaic.v1`` receipt over synthetic
  shards: crossing rate, KS statistic of the PIT histogram vs uniform,
  mean interval width.

Fail closed throughout: non-finite quantiles or outcomes raise; a tau grid
that disagrees with the emitted quantile count raises; an empty stream
raises. SYNTHETIC evidence only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.research.fleet_eval import (
    DEFAULT_TAUS,
    HeadFactory,
    ShardGenerator,
    resolve_shard_generators,
)
from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_revision

QUANTILE_MOSAIC_SCHEMA = "quantile_mosaic.v1"


def _check_taus(taus: Sequence[float]) -> tuple[float, ...]:
    if not taus:
        raise ValueError("taus must be non-empty")
    out = tuple(float(t) for t in taus)
    if not all(math.isfinite(t) for t in out):
        raise ValueError(f"taus must be finite, got {taus!r}")
    if any(t <= 0.0 or t >= 1.0 for t in out):
        raise ValueError(f"taus must lie in (0,1), got {taus!r}")
    if np.any(np.diff(np.asarray(out, dtype=float)) <= 0.0):
        raise ValueError("taus must be strictly increasing")
    return out


def _check_grid(q: NDArray[np.float64], n_taus: int) -> NDArray[np.float64]:
    arr = np.asarray(q, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != n_taus:
        raise ValueError(f"quantile grid must be (n, {n_taus}), got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("quantile grid must be finite")
    return arr


def pit_interpolated(
    taus: Sequence[float], quantiles_row: Sequence[float] | NDArray[np.float64], y: float
) -> float:
    """Piecewise-uniform PIT of ``y`` inside one row's quantile grid.

    Below the lowest quantile -> 0; above the highest -> 1; between
    bracketing knots the CDF is linear in y (equal-mass assumption between
    adjacent grid quantiles — the standard discrete-grid convention).
    """
    t = _check_taus(taus)
    q = np.asarray(quantiles_row, dtype=float)
    if q.ndim != 1 or q.size != len(t):
        raise ValueError(f"row must have {len(t)} quantiles, got shape {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError("quantile row must be finite")
    if not math.isfinite(y):
        raise ValueError(f"y must be finite, got {y!r}")
    # Monotone repair is NOT applied here: crossing is a defect the caller
    # audits — this function assumes a valid grid and uses searchsorted.
    if y <= q[0]:
        return 0.0
    if y >= q[-1]:
        return 1.0
    idx = int(np.searchsorted(q, y, side="right"))
    lo_q, hi_q = q[idx - 1], q[idx]
    lo_t, hi_t = t[idx - 1], t[idx]
    if hi_q <= lo_q:  # flat step in the grid (degenerate but valid)
        return float(hi_t)
    frac = (y - lo_q) / (hi_q - lo_q)
    return float(lo_t + frac * (hi_t - lo_t))


class QuantileMosaic:
    """Streaming PIT + per-tau breach accumulator for one quantile grid."""

    def __init__(self, taus: Sequence[float]) -> None:
        self.taus = _check_taus(taus)
        self.pits: list[float] = []
        self.n_rows = 0
        self.n_crossed_rows = 0
        self.breach_counts = np.zeros(len(self.taus), dtype=np.int64)

    def update(self, quantiles_row: Sequence[float], y: float) -> float:
        """Ingest one (grid row, outcome) pair; returns the PIT."""
        q = np.asarray(quantiles_row, dtype=float)
        if q.ndim != 1 or q.size != len(self.taus):
            raise ValueError(f"row must have {len(self.taus)} quantiles, got shape {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError("quantile row must be finite")
        if not math.isfinite(y):
            raise ValueError(f"y must be finite, got {y!r}")
        self.n_rows += 1
        if np.any(np.diff(q) < 0.0):
            self.n_crossed_rows += 1
        p = pit_interpolated(self.taus, q, y)
        self.pits.append(p)
        for j, qj in enumerate(q):
            if y > qj:
                self.breach_counts[j] += 1
        return p

    @property
    def crossing_rate(self) -> float:
        return float(self.n_crossed_rows / self.n_rows) if self.n_rows else float("nan")

    def breach_rates(self) -> NDArray[np.float64]:
        """Empirical P(y > q_tau) per tau — should equal 1 - tau."""
        if self.n_rows == 0:
            return np.full(len(self.taus), np.nan)
        return self.breach_counts.astype(float) / self.n_rows

    def pit_histogram(self, bins: int = 10) -> NDArray[np.float64]:
        """Normalized PIT histogram (density per bin, sums to 1)."""
        if isinstance(bins, bool) or int(bins) < 2:
            raise ValueError(f"bins must be an int >= 2, got {bins!r}")
        if not self.pits:
            raise ValueError("no observations")
        h, _ = np.histogram(np.asarray(self.pits), bins=int(bins), range=(0.0, 1.0))
        return h / h.sum()

    def pit_ks(self) -> float:
        """KS statistic of the empirical PIT distribution vs Uniform(0,1)."""
        if not self.pits:
            raise ValueError("no observations")
        p = np.sort(np.asarray(self.pits, dtype=float))
        n = p.size
        cdf = np.arange(1, n + 1) / n
        d_plus = float(np.max(cdf - p))
        d_minus = float(np.max(p - (np.arange(n) / n)))
        return max(d_plus, d_minus)


def quantile_consistency(
    quantiles: NDArray[np.float64],
    taus: Sequence[float],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Audit one (n, k) quantile grid for internal coherence.

    Returns crossing/width/skew diagnostics. ``strict=True`` raises on the
    first crossing row; ``strict=False`` counts violations instead.
    """
    t = _check_taus(taus)
    q = _check_grid(np.asarray(quantiles), len(t))
    diffs = np.diff(q, axis=1)
    crossed_rows = np.nonzero(np.any(diffs < 0.0, axis=1))[0]
    if strict and crossed_rows.size:
        raise ValueError(
            f"quantile crossing in {crossed_rows.size} rows (first at row {int(crossed_rows[0])})"
        )
    widths = q[:, -1] - q[:, 0]
    med_idx = int(np.argmin(np.abs(np.asarray(t) - 0.5)))
    mid = q[:, med_idx]
    mean_q = q.mean(axis=1)
    return {
        "n_rows": int(q.shape[0]),
        "n_crossed_rows": int(crossed_rows.size),
        "crossing_rate": float(crossed_rows.size / q.shape[0]),
        "min_interval_width": float(widths.min()),
        "mean_interval_width": float(widths.mean()),
        "skew_mid_minus_mean_mean": float((mid - mean_q).mean()),
        "taus": list(t),
    }


def mosaic_bench(
    head: HeadFactory,
    *,
    shard_names: Sequence[str] | None = None,
    taus: Sequence[float] = DEFAULT_TAUS,
    n: int = 512,
    seed: int = 0,
    bins: int = 10,
) -> dict[str, Any]:
    """Sealed ``quantile_mosaic.v1`` receipt over the resolved shards."""
    t = _check_taus(taus)
    if isinstance(n, bool) or int(n) < 8:
        raise ValueError(f"n must be an int >= 8, got {n!r}")
    gens = resolve_shard_generators(shard_names)
    if not gens:
        raise ValueError("no shard generators resolved")
    mosaic = QuantileMosaic(t)
    cons_rows = 0
    cons_crossed = 0
    width_sum = 0.0
    for name, gen in gens.items():
        shard: ShardGenerator = gen
        sh = shard(n, seed)
        y_all = np.asarray(sh.y, dtype=float)
        x_all = np.asarray(sh.x, dtype=float)
        n_train = n // 2
        model = head()
        model.fit(x_all[:n_train], y_all[:n_train])
        q = model.predict(x_all[n_train:])
        q = _check_grid(q, len(t))
        y = y_all[n_train:]
        if q.shape[0] != y.shape[0]:
            raise ValueError(f"shard {name}: {q.shape[0]} quantile rows vs {y.shape[0]} outcomes")
        cons = quantile_consistency(q, t, strict=False)
        cons_rows += cons["n_rows"]
        cons_crossed += cons["n_crossed_rows"]
        width_sum += cons["mean_interval_width"] * cons["n_rows"]
        for i in range(q.shape[0]):
            mosaic.update(q[i], float(y[i]))
    payload: dict[str, Any] = {
        "schema": QUANTILE_MOSAIC_SCHEMA,
        "kind": "quantile_mosaic",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "disclaimer": (
            "Intra-head quantile-grid consistency + interpolated PIT on "
            "synthetic shards; never market evidence."
        ),
        "taus": list(t),
        "shards": sorted(gens),
        "n_rows": int(mosaic.n_rows),
        "crossing_rate": float(cons_crossed / cons_rows),
        "mean_interval_width": float(width_sum / cons_rows),
        "pit_ks": float(mosaic.pit_ks()),
        "pit_histogram": mosaic.pit_histogram(bins).tolist(),
        "breach_rates": mosaic.breach_rates().tolist(),
    }
    payload["payload_sha256"] = hash_bytes(json.dumps(payload, sort_keys=True).encode())
    return payload


__all__ = [
    "QUANTILE_MOSAIC_SCHEMA",
    "QuantileMosaic",
    "mosaic_bench",
    "pit_interpolated",
    "quantile_consistency",
]
