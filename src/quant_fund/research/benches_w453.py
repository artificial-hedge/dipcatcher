"""Wave-453 higher-algebra-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bar_cobar import bench_bar_cobar
from quant_fund.models.deligne_conj import bench_deligne_conj
from quant_fund.models.factor_homology import bench_factor_homology
from quant_fund.models.hochschild_hom import bench_hochschild_hom
from quant_fund.models.operad_koszul import bench_operad_koszul
from quant_fund.models.primitive_elts import bench_primitive_elts

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


def bench_operad_koszul_family(seed: int = _SEED + 2624) -> dict[str, float]:
    return _floats(_finite_blob("operad_koszul", bench_operad_koszul(seed)))


def bench_bar_cobar_family(seed: int = _SEED + 2625) -> dict[str, float]:
    return _floats(_finite_blob("bar_cobar", bench_bar_cobar(seed)))


def bench_factor_homology_family(seed: int = _SEED + 2626) -> dict[str, float]:
    return _floats(_finite_blob("factor_homology", bench_factor_homology(seed)))


def bench_hochschild_hom_family(seed: int = _SEED + 2627) -> dict[str, float]:
    return _floats(_finite_blob("hochschild_hom", bench_hochschild_hom(seed)))


def bench_deligne_conj_family(seed: int = _SEED + 2628) -> dict[str, float]:
    return _floats(_finite_blob("deligne_conj", bench_deligne_conj(seed)))


def bench_primitive_elts_family(seed: int = _SEED + 2629) -> dict[str, float]:
    return _floats(_finite_blob("primitive_elts", bench_primitive_elts(seed)))
