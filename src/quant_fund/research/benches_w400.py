"""Wave-400 algebraic-geometry-7 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dual_ab_var import bench_dual_ab_var
from quant_fund.models.etale_cover import bench_etale_cover
from quant_fund.models.hom_stack_toy import bench_hom_stack_toy
from quant_fund.models.jacobian_toy import bench_jacobian_toy
from quant_fund.models.picard_variety import bench_picard_variety
from quant_fund.models.seesaw_theorem import bench_seesaw_theorem

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


def bench_etale_cover_family(seed: int = _SEED + 2306) -> dict[str, float]:
    return _floats(_finite_blob("etale_cover", bench_etale_cover(seed)))


def bench_jacobian_toy_family(seed: int = _SEED + 2307) -> dict[str, float]:
    return _floats(_finite_blob("jacobian_toy", bench_jacobian_toy(seed)))


def bench_hom_stack_toy_family(seed: int = _SEED + 2308) -> dict[str, float]:
    return _floats(_finite_blob("hom_stack_toy", bench_hom_stack_toy(seed)))


def bench_seesaw_theorem_family(
    seed: int = _SEED + 2309,
) -> dict[str, float]:
    return _floats(_finite_blob("seesaw_theorem", bench_seesaw_theorem(seed)))


def bench_picard_variety_family(
    seed: int = _SEED + 2310,
) -> dict[str, float]:
    return _floats(_finite_blob("picard_variety", bench_picard_variety(seed)))


def bench_dual_ab_var_family(seed: int = _SEED + 2311) -> dict[str, float]:
    return _floats(_finite_blob("dual_ab_var", bench_dual_ab_var(seed)))
