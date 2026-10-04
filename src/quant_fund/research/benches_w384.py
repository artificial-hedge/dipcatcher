"""Wave-384 group-theory-3 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.aut_group import bench_aut_group
from quant_fund.models.composition_series import bench_composition_series
from quant_fund.models.hall_subgroup import bench_hall_subgroup
from quant_fund.models.permutation_poly import bench_permutation_poly
from quant_fund.models.schur_multiplier import bench_schur_multiplier
from quant_fund.models.transfer_hom import bench_transfer_hom

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


def bench_hall_subgroup_family(seed: int = _SEED + 2210) -> dict[str, float]:
    return _floats(_finite_blob("hall_subgroup", bench_hall_subgroup(seed)))


def bench_transfer_hom_family(seed: int = _SEED + 2211) -> dict[str, float]:
    return _floats(_finite_blob("transfer_hom", bench_transfer_hom(seed)))


def bench_schur_multiplier_family(seed: int = _SEED + 2212) -> dict[str, float]:
    return _floats(_finite_blob("schur_multiplier", bench_schur_multiplier(seed)))


def bench_aut_group_family(seed: int = _SEED + 2213) -> dict[str, float]:
    return _floats(_finite_blob("aut_group", bench_aut_group(seed)))


def bench_composition_series_family(seed: int = _SEED + 2214) -> dict[str, float]:
    return _floats(_finite_blob("composition_series", bench_composition_series(seed)))


def bench_permutation_poly_family(seed: int = _SEED + 2215) -> dict[str, float]:
    return _floats(_finite_blob("permutation_poly", bench_permutation_poly(seed)))
