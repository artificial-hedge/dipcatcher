"""Wave-463 arithmetic-D-modules bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.arithmetic_dm import bench_arithmetic_dm
from quant_fund.models.frobenius_dm import bench_frobenius_dm
from quant_fund.models.holonomic_dm import bench_holonomic_dm
from quant_fund.models.isocrystal import bench_isocrystal
from quant_fund.models.overconv_dm import bench_overconv_dm
from quant_fund.models.rigid_dm import bench_rigid_dm

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


def bench_overconv_dm_family(seed: int = _SEED + 2684) -> dict[str, float]:
    return _floats(_finite_blob("overconv_dm", bench_overconv_dm(seed)))


def bench_arithmetic_dm_family(seed: int = _SEED + 2685) -> dict[str, float]:
    return _floats(_finite_blob("arithmetic_dm", bench_arithmetic_dm(seed)))


def bench_frobenius_dm_family(seed: int = _SEED + 2686) -> dict[str, float]:
    return _floats(_finite_blob("frobenius_dm", bench_frobenius_dm(seed)))


def bench_holonomic_dm_family(seed: int = _SEED + 2687) -> dict[str, float]:
    return _floats(_finite_blob("holonomic_dm", bench_holonomic_dm(seed)))


def bench_rigid_dm_family(seed: int = _SEED + 2688) -> dict[str, float]:
    return _floats(_finite_blob("rigid_dm", bench_rigid_dm(seed)))


def bench_isocrystal_family(seed: int = _SEED + 2689) -> dict[str, float]:
    return _floats(_finite_blob("isocrystal", bench_isocrystal(seed)))
