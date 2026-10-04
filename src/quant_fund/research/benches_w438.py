"""Wave-438 condensed-mathematics bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.analytic_ring import bench_analytic_ring
from quant_fund.models.condensed_set import bench_condensed_set
from quant_fund.models.light_condensed import bench_light_condensed
from quant_fund.models.liquid_group import bench_liquid_group
from quant_fund.models.proetale_site import bench_proetale_site
from quant_fund.models.solid_group import bench_solid_group

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


def bench_condensed_set_family(
    seed: int = _SEED + 2534,
) -> dict[str, float]:
    return _floats(_finite_blob("condensed_set", bench_condensed_set(seed)))


def bench_solid_group_family(
    seed: int = _SEED + 2535,
) -> dict[str, float]:
    return _floats(_finite_blob("solid_group", bench_solid_group(seed)))


def bench_liquid_group_family(
    seed: int = _SEED + 2536,
) -> dict[str, float]:
    return _floats(_finite_blob("liquid_group", bench_liquid_group(seed)))


def bench_proetale_site_family(
    seed: int = _SEED + 2537,
) -> dict[str, float]:
    return _floats(_finite_blob("proetale_site", bench_proetale_site(seed)))


def bench_light_condensed_family(
    seed: int = _SEED + 2538,
) -> dict[str, float]:
    return _floats(_finite_blob("light_condensed", bench_light_condensed(seed)))


def bench_analytic_ring_family(
    seed: int = _SEED + 2539,
) -> dict[str, float]:
    return _floats(_finite_blob("analytic_ring", bench_analytic_ring(seed)))
