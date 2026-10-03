"""Wave-675 higher-algebra-6 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.boards_operad import bench_boards_operad
from quant_fund.models.cyclotomic_e_n import bench_cyclotomic_e_n
from quant_fund.models.e3_algebra import bench_e3_algebra
from quant_fund.models.getzler_jones import bench_getzler_jones
from quant_fund.models.surfaces_operad import bench_surfaces_operad
from quant_fund.models.tadv_hochschild import bench_tadv_hochschild

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


def bench_e3_algebra_family(
    seed: int = _SEED + 6400,
) -> dict[str, float]:
    return _floats(_finite_blob("e3_algebra", bench_e3_algebra(seed)))


def bench_getzler_jones_family(
    seed: int = _SEED + 6401,
) -> dict[str, float]:
    return _floats(_finite_blob("getzler_jones", bench_getzler_jones(seed)))


def bench_tadv_hochschild_family(
    seed: int = _SEED + 6402,
) -> dict[str, float]:
    return _floats(_finite_blob("tadv_hochschild", bench_tadv_hochschild(seed)))


def bench_cyclotomic_e_n_family(
    seed: int = _SEED + 6403,
) -> dict[str, float]:
    return _floats(_finite_blob("cyclotomic_e_n", bench_cyclotomic_e_n(seed)))


def bench_surfaces_operad_family(
    seed: int = _SEED + 6404,
) -> dict[str, float]:
    return _floats(_finite_blob("surfaces_operad", bench_surfaces_operad(seed)))


def bench_boards_operad_family(
    seed: int = _SEED + 6405,
) -> dict[str, float]:
    return _floats(_finite_blob("boards_operad", bench_boards_operad(seed)))
