"""Wave-352 Lie-theory canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cartan_matrix import bench_cartan_matrix
from quant_fund.models.killing_form import bench_killing_form
from quant_fund.models.root_lattice_a2 import bench_root_lattice_a2
from quant_fund.models.sl2_structure import bench_sl2_structure
from quant_fund.models.su2_algebra import bench_su2_algebra
from quant_fund.models.weyl_group_a2 import bench_weyl_group_a2

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


def bench_cartan_matrix_family(seed: int = _SEED + 2019) -> dict[str, float]:
    return _floats(_finite_blob("cartan_matrix", bench_cartan_matrix(seed)))


def bench_weyl_group_a2_family(seed: int = _SEED + 2020) -> dict[str, float]:
    return _floats(_finite_blob("weyl_group_a2", bench_weyl_group_a2(seed)))


def bench_killing_form_family(seed: int = _SEED + 2021) -> dict[str, float]:
    return _floats(_finite_blob("killing_form", bench_killing_form(seed)))


def bench_root_lattice_a2_family(seed: int = _SEED + 2022) -> dict[str, float]:
    return _floats(_finite_blob("root_lattice_a2", bench_root_lattice_a2(seed)))


def bench_sl2_structure_family(seed: int = _SEED + 2023) -> dict[str, float]:
    return _floats(_finite_blob("sl2_structure", bench_sl2_structure(seed)))


def bench_su2_algebra_family(seed: int = _SEED + 2024) -> dict[str, float]:
    return _floats(_finite_blob("su2_algebra", bench_su2_algebra(seed)))
