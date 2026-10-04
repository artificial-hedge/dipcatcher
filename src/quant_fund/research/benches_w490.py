"""Wave-490 arithmetic-geometry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adelic_curve import bench_adelic_curve
from quant_fund.models.arakelov_deg import bench_arakelov_deg
from quant_fund.models.arith_rr import bench_arith_rr
from quant_fund.models.arithmetic_chow import bench_arithmetic_chow
from quant_fund.models.faltings_metric import bench_faltings_metric
from quant_fund.models.height_arakelov import bench_height_arakelov

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


def bench_arakelov_deg_family(seed: int = _SEED + 2846) -> dict[str, float]:
    return _floats(_finite_blob("arakelov_deg", bench_arakelov_deg(seed)))


def bench_adelic_curve_family(seed: int = _SEED + 2847) -> dict[str, float]:
    return _floats(_finite_blob("adelic_curve", bench_adelic_curve(seed)))


def bench_height_arakelov_family(seed: int = _SEED + 2848) -> dict[str, float]:
    return _floats(_finite_blob("height_arakelov", bench_height_arakelov(seed)))


def bench_faltings_metric_family(seed: int = _SEED + 2849) -> dict[str, float]:
    return _floats(_finite_blob("faltings_metric", bench_faltings_metric(seed)))


def bench_arithmetic_chow_family(seed: int = _SEED + 2850) -> dict[str, float]:
    return _floats(_finite_blob("arithmetic_chow", bench_arithmetic_chow(seed)))


def bench_arith_rr_family(seed: int = _SEED + 2851) -> dict[str, float]:
    return _floats(_finite_blob("arith_rr", bench_arith_rr(seed)))
