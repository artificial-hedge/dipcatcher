"""Wave-537 Riemannian-geometry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.comparison_thm import bench_comparison_thm
from quant_fund.models.jacobi_field import bench_jacobi_field
from quant_fund.models.levi_civita import bench_levi_civita
from quant_fund.models.ricci_scalar import bench_ricci_scalar
from quant_fund.models.riemann_curvature import bench_riemann_curvature
from quant_fund.models.riemann_metric import bench_riemann_metric

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


def bench_riemann_metric_family(seed: int = _SEED + 3128) -> dict[str, float]:
    return _floats(_finite_blob("riemann_metric", bench_riemann_metric(seed)))


def bench_levi_civita_family(seed: int = _SEED + 3129) -> dict[str, float]:
    return _floats(_finite_blob("levi_civita", bench_levi_civita(seed)))


def bench_riemann_curvature_family(
    seed: int = _SEED + 3130,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "riemann_curvature",
            bench_riemann_curvature(seed),
        )
    )


def bench_ricci_scalar_family(seed: int = _SEED + 3131) -> dict[str, float]:
    return _floats(_finite_blob("ricci_scalar", bench_ricci_scalar(seed)))


def bench_jacobi_field_family(seed: int = _SEED + 3132) -> dict[str, float]:
    return _floats(_finite_blob("jacobi_field", bench_jacobi_field(seed)))


def bench_comparison_thm_family(
    seed: int = _SEED + 3133,
) -> dict[str, float]:
    return _floats(_finite_blob("comparison_thm", bench_comparison_thm(seed)))
