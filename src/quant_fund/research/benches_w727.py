"""Wave-727 Galois-deformation-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.breuil_meizard import (
    bench_breuil_meizard,
)
from quant_fund.models.caruso_lebaron import (
    bench_caruso_lebaron,
)
from quant_fund.models.galdef_ring import bench_galdef_ring
from quant_fund.models.gee_kisin import bench_gee_kisin
from quant_fund.models.patching_arg import bench_patching_arg
from quant_fund.models.taylor_wiles import bench_taylor_wiles

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


def bench_galdef_ring_family(
    seed: int = _SEED + 11600,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "galdef_ring",
            bench_galdef_ring(seed),
        )
    )


def bench_patching_arg_family(
    seed: int = _SEED + 11601,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "patching_arg",
            bench_patching_arg(seed),
        )
    )


def bench_taylor_wiles_family(
    seed: int = _SEED + 11602,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "taylor_wiles",
            bench_taylor_wiles(seed),
        )
    )


def bench_breuil_meizard_family(
    seed: int = _SEED + 11603,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "breuil_meizard",
            bench_breuil_meizard(seed),
        )
    )


def bench_gee_kisin_family(
    seed: int = _SEED + 11604,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gee_kisin",
            bench_gee_kisin(seed),
        )
    )


def bench_caruso_lebaron_family(
    seed: int = _SEED + 11605,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "caruso_lebaron",
            bench_caruso_lebaron(seed),
        )
    )
