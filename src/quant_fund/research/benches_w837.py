"""Wave-837 convex-geometry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.alexandrov_fenchel import (
    bench_alexandrov_fenchel,
)
from quant_fund.models.brunn_minkowski import (
    bench_brunn_minkowski,
)
from quant_fund.models.helly_theorem import (
    bench_helly_theorem,
)
from quant_fund.models.isoperimetric_ineq import (
    bench_isoperimetric_ineq,
)
from quant_fund.models.minkowski_sum import (
    bench_minkowski_sum,
)
from quant_fund.models.mixed_volume import (
    bench_mixed_volume,
)

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


def bench_brunn_minkowski_family(
    seed: int = _SEED + 22500,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "brunn_minkowski",
            bench_brunn_minkowski(seed),
        )
    )


def bench_alexandrov_fenchel_family(
    seed: int = _SEED + 22501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "alexandrov_fenchel",
            bench_alexandrov_fenchel(seed),
        )
    )


def bench_isoperimetric_ineq_family(
    seed: int = _SEED + 22502,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "isoperimetric_ineq",
            bench_isoperimetric_ineq(seed),
        )
    )


def bench_minkowski_sum_family(
    seed: int = _SEED + 22503,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "minkowski_sum",
            bench_minkowski_sum(seed),
        )
    )


def bench_mixed_volume_family(
    seed: int = _SEED + 22504,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mixed_volume",
            bench_mixed_volume(seed),
        )
    )


def bench_helly_theorem_family(
    seed: int = _SEED + 22505,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "helly_theorem",
            bench_helly_theorem(seed),
        )
    )
