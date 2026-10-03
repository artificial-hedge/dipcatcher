"""Wave-530 nonuniform-hyperbolicity bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dominated_split import bench_dominated_split
from quant_fund.models.katok_horseshoe import bench_katok_horseshoe
from quant_fund.models.lyapunov_chart import bench_lyapunov_chart
from quant_fund.models.nonuniform_hyp import bench_nonuniform_hyp
from quant_fund.models.osceledets_reg import bench_osceledets_reg
from quant_fund.models.pesin_theory import bench_pesin_theory

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


def bench_pesin_theory_family(seed: int = _SEED + 3086) -> dict[str, float]:
    return _floats(_finite_blob("pesin_theory", bench_pesin_theory(seed)))


def bench_nonuniform_hyp_family(seed: int = _SEED + 3087) -> dict[str, float]:
    return _floats(_finite_blob("nonuniform_hyp", bench_nonuniform_hyp(seed)))


def bench_dominated_split_family(seed: int = _SEED + 3088) -> dict[str, float]:
    return _floats(_finite_blob("dominated_split", bench_dominated_split(seed)))


def bench_osceledets_reg_family(seed: int = _SEED + 3089) -> dict[str, float]:
    return _floats(_finite_blob("osceledets_reg", bench_osceledets_reg(seed)))


def bench_lyapunov_chart_family(seed: int = _SEED + 3090) -> dict[str, float]:
    return _floats(_finite_blob("lyapunov_chart", bench_lyapunov_chart(seed)))


def bench_katok_horseshoe_family(seed: int = _SEED + 3091) -> dict[str, float]:
    return _floats(_finite_blob("katok_horseshoe", bench_katok_horseshoe(seed)))
