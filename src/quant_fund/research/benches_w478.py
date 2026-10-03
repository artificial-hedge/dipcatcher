"""Wave-478 motivic-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.beilinson_con import bench_beilinson_con
from quant_fund.models.cellular_motive import bench_cellular_motive
from quant_fund.models.levine_morel import bench_levine_morel
from quant_fund.models.mgl_spec import bench_mgl_spec
from quant_fund.models.motivic_pi0 import bench_motivic_pi0
from quant_fund.models.quadratic_k import bench_quadratic_k

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


def bench_levine_morel_family(seed: int = _SEED + 2774) -> dict[str, float]:
    return _floats(_finite_blob("levine_morel", bench_levine_morel(seed)))


def bench_quadratic_k_family(seed: int = _SEED + 2775) -> dict[str, float]:
    return _floats(_finite_blob("quadratic_k", bench_quadratic_k(seed)))


def bench_mgl_spec_family(seed: int = _SEED + 2776) -> dict[str, float]:
    return _floats(_finite_blob("mgl_spec", bench_mgl_spec(seed)))


def bench_cellular_motive_family(seed: int = _SEED + 2777) -> dict[str, float]:
    return _floats(_finite_blob("cellular_motive", bench_cellular_motive(seed)))


def bench_motivic_pi0_family(seed: int = _SEED + 2778) -> dict[str, float]:
    return _floats(_finite_blob("motivic_pi0", bench_motivic_pi0(seed)))


def bench_beilinson_con_family(seed: int = _SEED + 2779) -> dict[str, float]:
    return _floats(_finite_blob("beilinson_con", bench_beilinson_con(seed)))
