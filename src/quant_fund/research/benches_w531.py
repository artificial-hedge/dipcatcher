"""Wave-531 bifurcation-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bogdanov_takens import bench_bogdanov_takens
from quant_fund.models.homoclinic_bif import bench_homoclinic_bif
from quant_fund.models.hopf_bif import bench_hopf_bif
from quant_fund.models.neimark_sacker import bench_neimark_sacker
from quant_fund.models.period_doubling import bench_period_doubling
from quant_fund.models.saddle_node import bench_saddle_node

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


def bench_saddle_node_family(seed: int = _SEED + 3092) -> dict[str, float]:
    return _floats(_finite_blob("saddle_node", bench_saddle_node(seed)))


def bench_hopf_bif_family(seed: int = _SEED + 3093) -> dict[str, float]:
    return _floats(_finite_blob("hopf_bif", bench_hopf_bif(seed)))


def bench_period_doubling_family(seed: int = _SEED + 3094) -> dict[str, float]:
    return _floats(_finite_blob("period_doubling", bench_period_doubling(seed)))


def bench_neimark_sacker_family(seed: int = _SEED + 3095) -> dict[str, float]:
    return _floats(_finite_blob("neimark_sacker", bench_neimark_sacker(seed)))


def bench_bogdanov_takens_family(
    seed: int = _SEED + 3096,
) -> dict[str, float]:
    return _floats(_finite_blob("bogdanov_takens", bench_bogdanov_takens(seed)))


def bench_homoclinic_bif_family(seed: int = _SEED + 3097) -> dict[str, float]:
    return _floats(_finite_blob("homoclinic_bif", bench_homoclinic_bif(seed)))
