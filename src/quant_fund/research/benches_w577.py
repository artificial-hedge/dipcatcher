"""Wave-577 perverse-sheaves bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.char_cycle import bench_char_cycle
from quant_fund.models.d_module2 import bench_d_module2
from quant_fund.models.intersection_homology import (
    bench_intersection_homology,
)
from quant_fund.models.middle_perversity import (
    bench_middle_perversity,
)
from quant_fund.models.nearby_cycles import bench_nearby_cycles
from quant_fund.models.perverse_sheaf import (
    bench_perverse_sheaf,
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


def bench_perverse_sheaf_family(
    seed: int = _SEED + 3368,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "perverse_sheaf",
            bench_perverse_sheaf(seed),
        )
    )


def bench_intersection_homology_family(
    seed: int = _SEED + 3369,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "intersection_homology",
            bench_intersection_homology(seed),
        )
    )


def bench_nearby_cycles_family(
    seed: int = _SEED + 3370,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nearby_cycles",
            bench_nearby_cycles(seed),
        )
    )


def bench_d_module2_family(seed: int = _SEED + 3371) -> dict[str, float]:
    return _floats(_finite_blob("d_module2", bench_d_module2(seed)))


def bench_char_cycle_family(seed: int = _SEED + 3372) -> dict[str, float]:
    return _floats(_finite_blob("char_cycle", bench_char_cycle(seed)))


def bench_middle_perversity_family(
    seed: int = _SEED + 3373,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "middle_perversity",
            bench_middle_perversity(seed),
        )
    )
