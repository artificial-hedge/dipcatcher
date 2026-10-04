"""Wave-494 noncommutative-geometry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.calabi_yau_alg import bench_calabi_yau_alg
from quant_fund.models.connes_nc import bench_connes_nc
from quant_fund.models.cyclic_coh import bench_cyclic_coh
from quant_fund.models.ginzburg_dga import bench_ginzburg_dga
from quant_fund.models.hochschild_coh import bench_hochschild_coh
from quant_fund.models.nc_scheme import bench_nc_scheme

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


def bench_hochschild_coh_family(seed: int = _SEED + 2870) -> dict[str, float]:
    return _floats(_finite_blob("hochschild_coh", bench_hochschild_coh(seed)))


def bench_cyclic_coh_family(seed: int = _SEED + 2871) -> dict[str, float]:
    return _floats(_finite_blob("cyclic_coh", bench_cyclic_coh(seed)))


def bench_nc_scheme_family(seed: int = _SEED + 2872) -> dict[str, float]:
    return _floats(_finite_blob("nc_scheme", bench_nc_scheme(seed)))


def bench_calabi_yau_alg_family(seed: int = _SEED + 2873) -> dict[str, float]:
    return _floats(_finite_blob("calabi_yau_alg", bench_calabi_yau_alg(seed)))


def bench_ginzburg_dga_family(seed: int = _SEED + 2874) -> dict[str, float]:
    return _floats(_finite_blob("ginzburg_dga", bench_ginzburg_dga(seed)))


def bench_connes_nc_family(seed: int = _SEED + 2875) -> dict[str, float]:
    return _floats(_finite_blob("connes_nc", bench_connes_nc(seed)))
