"""Wave-362 complex-analysis canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.argument_principle import bench_argument_principle
from quant_fund.models.cauchy_integral import bench_cauchy_integral
from quant_fund.models.conformal_map import bench_conformal_map
from quant_fund.models.laurent_series import bench_laurent_series
from quant_fund.models.liouville import bench_liouville
from quant_fund.models.residue_calc import bench_residue_calc

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


def bench_cauchy_integral_family(seed: int = _SEED + 2079) -> dict[str, float]:
    return _floats(_finite_blob("cauchy_integral", bench_cauchy_integral(seed)))


def bench_residue_calc_family(seed: int = _SEED + 2080) -> dict[str, float]:
    return _floats(_finite_blob("residue_calc", bench_residue_calc(seed)))


def bench_laurent_series_family(seed: int = _SEED + 2081) -> dict[str, float]:
    return _floats(_finite_blob("laurent_series", bench_laurent_series(seed)))


def bench_argument_principle_family(seed: int = _SEED + 2082) -> dict[str, float]:
    return _floats(_finite_blob("argument_principle", bench_argument_principle(seed)))


def bench_conformal_map_family(seed: int = _SEED + 2083) -> dict[str, float]:
    return _floats(_finite_blob("conformal_map", bench_conformal_map(seed)))


def bench_liouville_family(seed: int = _SEED + 2084) -> dict[str, float]:
    return _floats(_finite_blob("liouville", bench_liouville(seed)))
