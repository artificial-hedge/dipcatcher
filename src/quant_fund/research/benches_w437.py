"""Wave-437 p-adic cohomology bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.comparison_iso import bench_comparison_iso
from quant_fund.models.crystalline_coh import bench_crystalline_coh
from quant_fund.models.derham_coh import bench_derham_coh
from quant_fund.models.etale_coh import bench_etale_coh
from quant_fund.models.frobenius_coh import bench_frobenius_coh
from quant_fund.models.prismatic_coh import bench_prismatic_coh

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


def bench_crystalline_coh_family(
    seed: int = _SEED + 2528,
) -> dict[str, float]:
    return _floats(_finite_blob("crystalline_coh", bench_crystalline_coh(seed)))


def bench_prismatic_coh_family(
    seed: int = _SEED + 2529,
) -> dict[str, float]:
    return _floats(_finite_blob("prismatic_coh", bench_prismatic_coh(seed)))


def bench_etale_coh_family(
    seed: int = _SEED + 2530,
) -> dict[str, float]:
    return _floats(_finite_blob("etale_coh", bench_etale_coh(seed)))


def bench_derham_coh_family(
    seed: int = _SEED + 2531,
) -> dict[str, float]:
    return _floats(_finite_blob("derham_coh", bench_derham_coh(seed)))


def bench_frobenius_coh_family(
    seed: int = _SEED + 2532,
) -> dict[str, float]:
    return _floats(_finite_blob("frobenius_coh", bench_frobenius_coh(seed)))


def bench_comparison_iso_family(
    seed: int = _SEED + 2533,
) -> dict[str, float]:
    return _floats(_finite_blob("comparison_iso", bench_comparison_iso(seed)))
