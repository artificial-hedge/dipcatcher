"""Wave-402 topos-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.etale_space import bench_etale_space
from quant_fund.models.geometric_morph import bench_geometric_morph
from quant_fund.models.groth_topo import bench_groth_topo
from quant_fund.models.logic_topos import bench_logic_topos
from quant_fund.models.sheaf_cond import bench_sheaf_cond
from quant_fund.models.topos_subobj import bench_topos_subobj

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


def bench_topos_subobj_family(seed: int = _SEED + 2318) -> dict[str, float]:
    return _floats(_finite_blob("topos_subobj", bench_topos_subobj(seed)))


def bench_groth_topo_family(seed: int = _SEED + 2319) -> dict[str, float]:
    return _floats(_finite_blob("groth_topo", bench_groth_topo(seed)))


def bench_sheaf_cond_family(seed: int = _SEED + 2320) -> dict[str, float]:
    return _floats(_finite_blob("sheaf_cond", bench_sheaf_cond(seed)))


def bench_logic_topos_family(seed: int = _SEED + 2321) -> dict[str, float]:
    return _floats(_finite_blob("logic_topos", bench_logic_topos(seed)))


def bench_geometric_morph_family(
    seed: int = _SEED + 2322,
) -> dict[str, float]:
    return _floats(_finite_blob("geometric_morph", bench_geometric_morph(seed)))


def bench_etale_space_family(seed: int = _SEED + 2323) -> dict[str, float]:
    return _floats(_finite_blob("etale_space", bench_etale_space(seed)))
