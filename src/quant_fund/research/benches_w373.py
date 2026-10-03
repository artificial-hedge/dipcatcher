"""Wave-373 optimization-3 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bfgs_wolfe import bench_bfgs_wolfe
from quant_fund.models.bundle_method import bench_bundle_method
from quant_fund.models.frank_wolfe2 import bench_frank_wolfe2
from quant_fund.models.ip_qp import bench_ip_qp
from quant_fund.models.sqp import bench_sqp
from quant_fund.models.trust_region import bench_trust_region

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


def bench_bundle_method_family(seed: int = _SEED + 2144) -> dict[str, float]:
    return _floats(_finite_blob("bundle_method", bench_bundle_method(seed)))


def bench_sqp_family(seed: int = _SEED + 2145) -> dict[str, float]:
    return _floats(_finite_blob("sqp", bench_sqp(seed)))


def bench_ip_qp_family(seed: int = _SEED + 2146) -> dict[str, float]:
    return _floats(_finite_blob("ip_qp", bench_ip_qp(seed)))


def bench_trust_region_family(seed: int = _SEED + 2147) -> dict[str, float]:
    return _floats(_finite_blob("trust_region", bench_trust_region(seed)))


def bench_frank_wolfe2_family(seed: int = _SEED + 2148) -> dict[str, float]:
    return _floats(_finite_blob("frank_wolfe2", bench_frank_wolfe2(seed)))


def bench_bfgs_wolfe_family(seed: int = _SEED + 2149) -> dict[str, float]:
    return _floats(_finite_blob("bfgs_wolfe", bench_bfgs_wolfe(seed)))
