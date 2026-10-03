"""Wave-481 motivic-5 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.fqmotive import bench_fqmotive
from quant_fund.models.higher_chow2 import bench_higher_chow2
from quant_fund.models.motivic_chern import bench_motivic_chern
from quant_fund.models.motivic_landin import bench_motivic_landin
from quant_fund.models.mtc_motive import bench_mtc_motive
from quant_fund.models.triang_motive import bench_triang_motive

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


def bench_mtc_motive_family(seed: int = _SEED + 2792) -> dict[str, float]:
    return _floats(_finite_blob("mtc_motive", bench_mtc_motive(seed)))


def bench_fqmotive_family(seed: int = _SEED + 2793) -> dict[str, float]:
    return _floats(_finite_blob("fqmotive", bench_fqmotive(seed)))


def bench_triang_motive_family(seed: int = _SEED + 2794) -> dict[str, float]:
    return _floats(_finite_blob("triang_motive", bench_triang_motive(seed)))


def bench_motivic_chern_family(seed: int = _SEED + 2795) -> dict[str, float]:
    return _floats(_finite_blob("motivic_chern", bench_motivic_chern(seed)))


def bench_motivic_landin_family(seed: int = _SEED + 2796) -> dict[str, float]:
    return _floats(_finite_blob("motivic_landin", bench_motivic_landin(seed)))


def bench_higher_chow2_family(seed: int = _SEED + 2797) -> dict[str, float]:
    return _floats(_finite_blob("higher_chow2", bench_higher_chow2(seed)))
