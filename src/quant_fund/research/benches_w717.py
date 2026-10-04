"""Wave-717 noncommutative-motives bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bondal_kapranov import (
    bench_bondal_kapranov,
)
from quant_fund.models.dg_enhancement import (
    bench_dg_enhancement,
)
from quant_fund.models.enhanced_triangulated import (
    bench_enhanced_triangulated,
)
from quant_fund.models.nc_k_theory import bench_nc_k_theory
from quant_fund.models.nc_motive import bench_nc_motive
from quant_fund.models.tabuada_motive import (
    bench_tabuada_motive,
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


def bench_nc_motive_family(
    seed: int = _SEED + 10600,
) -> dict[str, float]:
    return _floats(_finite_blob("nc_motive", bench_nc_motive(seed)))


def bench_dg_enhancement_family(
    seed: int = _SEED + 10601,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dg_enhancement",
            bench_dg_enhancement(seed),
        )
    )


def bench_bondal_kapranov_family(
    seed: int = _SEED + 10602,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bondal_kapranov",
            bench_bondal_kapranov(seed),
        )
    )


def bench_enhanced_triangulated_family(
    seed: int = _SEED + 10603,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "enhanced_triangulated",
            bench_enhanced_triangulated(seed),
        )
    )


def bench_tabuada_motive_family(
    seed: int = _SEED + 10604,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tabuada_motive",
            bench_tabuada_motive(seed),
        )
    )


def bench_nc_k_theory_family(
    seed: int = _SEED + 10605,
) -> dict[str, float]:
    return _floats(_finite_blob("nc_k_theory", bench_nc_k_theory(seed)))
