"""Wave-439 formal-groups/chromatic bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.formal_group import bench_formal_group
from quant_fund.models.formal_module import bench_formal_module
from quant_fund.models.height_strata import bench_height_strata
from quant_fund.models.lazard_ring import bench_lazard_ring
from quant_fund.models.lubin_tate import bench_lubin_tate
from quant_fund.models.morava_k import bench_morava_k

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


def bench_formal_group_family(
    seed: int = _SEED + 2540,
) -> dict[str, float]:
    return _floats(_finite_blob("formal_group", bench_formal_group(seed)))


def bench_lazard_ring_family(
    seed: int = _SEED + 2541,
) -> dict[str, float]:
    return _floats(_finite_blob("lazard_ring", bench_lazard_ring(seed)))


def bench_formal_module_family(
    seed: int = _SEED + 2542,
) -> dict[str, float]:
    return _floats(_finite_blob("formal_module", bench_formal_module(seed)))


def bench_height_strata_family(
    seed: int = _SEED + 2543,
) -> dict[str, float]:
    return _floats(_finite_blob("height_strata", bench_height_strata(seed)))


def bench_lubin_tate_family(
    seed: int = _SEED + 2544,
) -> dict[str, float]:
    return _floats(_finite_blob("lubin_tate", bench_lubin_tate(seed)))


def bench_morava_k_family(
    seed: int = _SEED + 2545,
) -> dict[str, float]:
    return _floats(_finite_blob("morava_k", bench_morava_k(seed)))
