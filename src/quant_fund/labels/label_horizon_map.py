"""Label-parameter sensitivity surface (AFML ch. 3 diagnostics).

Triple-barrier labels depend on three free knobs — barrier width
(``pt``/``sl``), horizon, and the cUSUM event threshold ``h`` — and the
meta-labeling layer is only meaningful where the label is *non-degenerate*:
positive-rate ~0 or ~1 means the classifier learns the base rate, not the
signal. ``label_horizon_map`` sweeps the grid on a price path and reports,
per (barrier, horizon) cell:

- ``positive_rate`` — share of labeled events with label +1
- ``event_rate`` — share of bars flagged as events (cusum count / n)
- ``mean_horizon_used`` — mean bars until first touch / vertical barrier
- ``n_unobservable`` — events with no forward path (final-bar honesty)

The bench sweeps a synthetic GBM tape and a regime-switching tape and
reports which cells sit inside the [0.2, 0.8] positive-rate band where a
meta-model can actually earn its keep.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.labels.barriers import cusum_filter, triple_barrier
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]


def label_horizon_map(
    close: Array,
    *,
    barrier_widths: tuple[float, ...] = (0.005, 0.01, 0.02, 0.04),
    horizons: tuple[int, ...] = (5, 10, 20, 40),
    cusum_h: float = 0.0,
    min_ret: float = 0.0,
) -> list[dict[str, Any]]:
    """Sweep (pt=sl=width, horizon) cells on a close-price path."""
    c = np.asarray(close, dtype=float)
    if not np.all(np.isfinite(c)) or c.size < 10 or np.any(c <= 0):
        raise ValueError("close must be finite, positive, len>=10")
    events = (
        np.arange(c.size - 1, dtype=np.intp)
        if cusum_h <= 0
        else cusum_filter(np.diff(np.log(c)), cusum_h)
    )
    cells: list[dict[str, Any]] = []
    for w in barrier_widths:
        for h in horizons:
            if w <= 0 or h <= 0:
                raise ValueError("barrier widths and horizons must be positive")
            out = triple_barrier(c, events, w, w, h, min_ret=min_ret)
            labels = out["label"]
            finite = np.isfinite(labels)
            n = int(finite.sum())
            touches = out["t_touch"]
            cells.append(
                {
                    "barrier_width": float(w),
                    "horizon": int(h),
                    "n_events": int(events.size),
                    "n_labeled": n,
                    "n_unobservable": int(events.size - n),
                    "event_rate": float(events.size) / float(c.size),
                    "positive_rate": (float((labels[finite] > 0).sum()) / float(n) if n else None),
                    "mean_bars_to_touch": (
                        float(np.mean(touches[finite] - events[finite])) if n else None
                    ),
                }
            )
    return cells


def _gbm(seed: int, n: int, s0: float = 100.0, vol: float = 0.002) -> Array:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, vol, n)
    return s0 * np.exp(np.cumsum(r))


def _regime_gbm(seed: int, n: int) -> Array:
    rng = np.random.default_rng(seed)
    vols = np.where(rng.random(n) < 0.2, 0.006, 0.0015)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, vols)))


def label_horizon_map_bench(seed: int = 0, n_bars: int = 4000) -> dict[str, Any]:
    """Sweep both tapes; report the viable-cell share in the [0.2, 0.8] band."""
    tapes = {"gbm": _gbm(seed, n_bars), "regime": _regime_gbm(seed + 1, n_bars)}
    per_tape: dict[str, Any] = {}
    viable = 0
    total = 0
    for name, c in tapes.items():
        cells = label_horizon_map(c)
        for cell in cells:
            total += 1
            pr = cell["positive_rate"]
            ok = pr is not None and 0.2 <= pr <= 0.8
            cell["viable"] = ok
            viable += int(ok)
        per_tape[name] = cells
    payload: dict[str, Any] = {
        "kind": "label_horizon_map",
        "schema": "label_horizon_map.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "viable_band": [0.2, 0.8],
            "viable_cells": viable,
            "total_cells": total,
            "viable_share": viable / total if total else None,
            "verdict": "ok" if viable >= max(1, total // 2) else "weak",
        },
        "interpretation": {
            "grid": "barrier_widths x horizons, symmetric pt=sl, cusum_h=0 (all bars)",
            "per_tape": per_tape,
        },
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
