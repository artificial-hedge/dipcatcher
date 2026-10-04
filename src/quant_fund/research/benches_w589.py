"""Wave-589 topos-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cartesian_closed import (
    bench_cartesian_closed,
)
from quant_fund.models.coherent_topos import bench_coherent_topos
from quant_fund.models.internal_logic import (
    bench_internal_logic,
)
from quant_fund.models.power_object import bench_power_object
from quant_fund.models.pretopos import bench_pretopos
from quant_fund.models.subobject_lattice import (
    bench_subobject_lattice,
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


def bench_cartesian_closed_family(
    seed: int = _SEED + 3440,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cartesian_closed",
            bench_cartesian_closed(seed),
        )
    )


def bench_internal_logic_family(
    seed: int = _SEED + 3441,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "internal_logic",
            bench_internal_logic(seed),
        )
    )


def bench_subobject_lattice_family(
    seed: int = _SEED + 3442,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "subobject_lattice",
            bench_subobject_lattice(seed),
        )
    )


def bench_power_object_family(
    seed: int = _SEED + 3443,
) -> dict[str, float]:
    return _floats(_finite_blob("power_object", bench_power_object(seed)))


def bench_pretopos_family(seed: int = _SEED + 3444) -> dict[str, float]:
    return _floats(_finite_blob("pretopos", bench_pretopos(seed)))


def bench_coherent_topos_family(
    seed: int = _SEED + 3445,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "coherent_topos",
            bench_coherent_topos(seed),
        )
    )
