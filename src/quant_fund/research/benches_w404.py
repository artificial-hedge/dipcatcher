"""Wave-404 operad-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.brace_operad import bench_brace_operad
from quant_fund.models.little_intervals import bench_little_intervals
from quant_fund.models.operad_algt import bench_operad_algt
from quant_fund.models.operad_homology import bench_operad_homology
from quant_fund.models.props_toy import bench_props_toy
from quant_fund.models.swiss_cheese import bench_swiss_cheese

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            f = float(arr)
            if not np.isfinite(f):
                raise ValueError(f"non-finite {k} in {name}")
            flat[k] = f
        else:
            for i, val in enumerate(arr.ravel()):
                f = float(val)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite {k}[{i}] in {name}")
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_operad_algt_family(seed: int = _SEED + 2330) -> dict[str, float]:
    return _floats(_finite_blob("operad_algt", bench_operad_algt(seed)))


def bench_brace_operad_family(seed: int = _SEED + 2331) -> dict[str, float]:
    return _floats(_finite_blob("brace_operad", bench_brace_operad(seed)))


def bench_swiss_cheese_family(seed: int = _SEED + 2332) -> dict[str, float]:
    return _floats(_finite_blob("swiss_cheese", bench_swiss_cheese(seed)))


def bench_little_intervals_family(
    seed: int = _SEED + 2333,
) -> dict[str, float]:
    return _floats(_finite_blob("little_intervals", bench_little_intervals(seed)))


def bench_operad_homology_family(
    seed: int = _SEED + 2334,
) -> dict[str, float]:
    return _floats(_finite_blob("operad_homology", bench_operad_homology(seed)))


def bench_props_toy_family(seed: int = _SEED + 2335) -> dict[str, float]:
    return _floats(_finite_blob("props_toy", bench_props_toy(seed)))
