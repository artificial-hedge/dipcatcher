"""Wave-610 deformations-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.first_order import bench_first_order
from quant_fund.models.obstruction_def import (
    bench_obstruction_def,
)
from quant_fund.models.prorepresent import (
    bench_prorepresent,
)
from quant_fund.models.schlessinger2 import (
    bench_schlessinger2,
)
from quant_fund.models.semiuniversal import (
    bench_semiuniversal,
)
from quant_fund.models.versal_def import bench_versal_def

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


def bench_schlessinger2_family(
    seed: int = _SEED + 3566,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "schlessinger2",
            bench_schlessinger2(seed),
        )
    )


def bench_prorepresent_family(
    seed: int = _SEED + 3567,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "prorepresent",
            bench_prorepresent(seed),
        )
    )


def bench_versal_def_family(seed: int = _SEED + 3568) -> dict[str, float]:
    return _floats(_finite_blob("versal_def", bench_versal_def(seed)))


def bench_semiuniversal_family(
    seed: int = _SEED + 3569,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "semiuniversal",
            bench_semiuniversal(seed),
        )
    )


def bench_first_order_family(
    seed: int = _SEED + 3570,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "first_order",
            bench_first_order(seed),
        )
    )


def bench_obstruction_def_family(
    seed: int = _SEED + 3571,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "obstruction_def",
            bench_obstruction_def(seed),
        )
    )
