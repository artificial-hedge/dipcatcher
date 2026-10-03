"""Wave-474 TQFT-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cobordism_hyp import bench_cobordism_hyp
from quant_fund.models.heegaard_floer import bench_heegaard_floer
from quant_fund.models.khovanov import bench_khovanov
from quant_fund.models.modular_cat import bench_modular_cat
from quant_fund.models.reshet_turaev import bench_reshet_turaev
from quant_fund.models.topological_order import bench_topological_order

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


def bench_reshet_turaev_family(seed: int = _SEED + 2750) -> dict[str, float]:
    return _floats(_finite_blob("reshet_turaev", bench_reshet_turaev(seed)))


def bench_khovanov_family(seed: int = _SEED + 2751) -> dict[str, float]:
    return _floats(_finite_blob("khovanov", bench_khovanov(seed)))


def bench_heegaard_floer_family(seed: int = _SEED + 2752) -> dict[str, float]:
    return _floats(_finite_blob("heegaard_floer", bench_heegaard_floer(seed)))


def bench_cobordism_hyp_family(seed: int = _SEED + 2753) -> dict[str, float]:
    return _floats(_finite_blob("cobordism_hyp", bench_cobordism_hyp(seed)))


def bench_modular_cat_family(seed: int = _SEED + 2754) -> dict[str, float]:
    return _floats(_finite_blob("modular_cat", bench_modular_cat(seed)))


def bench_topological_order_family(seed: int = _SEED + 2755) -> dict[str, float]:
    return _floats(_finite_blob("topological_order", bench_topological_order(seed)))
