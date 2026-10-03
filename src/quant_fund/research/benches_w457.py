"""Wave-457 double-category/proarrow bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.companion_conj import bench_companion_conj
from quant_fund.models.fibrant_double import bench_fibrant_double
from quant_fund.models.framed_bicat import bench_framed_bicat
from quant_fund.models.proarrow import bench_proarrow
from quant_fund.models.tabulation import bench_tabulation
from quant_fund.models.virtual_equip import bench_virtual_equip

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


def bench_proarrow_family(seed: int = _SEED + 2648) -> dict[str, float]:
    return _floats(_finite_blob("proarrow", bench_proarrow(seed)))


def bench_virtual_equip_family(seed: int = _SEED + 2649) -> dict[str, float]:
    return _floats(_finite_blob("virtual_equip", bench_virtual_equip(seed)))


def bench_fibrant_double_family(seed: int = _SEED + 2650) -> dict[str, float]:
    return _floats(_finite_blob("fibrant_double", bench_fibrant_double(seed)))


def bench_tabulation_family(seed: int = _SEED + 2651) -> dict[str, float]:
    return _floats(_finite_blob("tabulation", bench_tabulation(seed)))


def bench_companion_conj_family(seed: int = _SEED + 2652) -> dict[str, float]:
    return _floats(_finite_blob("companion_conj", bench_companion_conj(seed)))


def bench_framed_bicat_family(seed: int = _SEED + 2653) -> dict[str, float]:
    return _floats(_finite_blob("framed_bicat", bench_framed_bicat(seed)))
