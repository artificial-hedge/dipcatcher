"""Wave-502 dg-category bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dg_cat2 import bench_dg_cat2
from quant_fund.models.dg_morita import bench_dg_morita
from quant_fund.models.dg_nerve import bench_dg_nerve
from quant_fund.models.dg_quotient import bench_dg_quotient
from quant_fund.models.drinfeld_quotient import bench_drinfeld_quotient
from quant_fund.models.keller_dg import bench_keller_dg

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


def bench_dg_cat2_family(seed: int = _SEED + 2918) -> dict[str, float]:
    return _floats(_finite_blob("dg_cat2", bench_dg_cat2(seed)))


def bench_dg_morita_family(seed: int = _SEED + 2919) -> dict[str, float]:
    return _floats(_finite_blob("dg_morita", bench_dg_morita(seed)))


def bench_dg_quotient_family(seed: int = _SEED + 2920) -> dict[str, float]:
    return _floats(_finite_blob("dg_quotient", bench_dg_quotient(seed)))


def bench_drinfeld_quotient_family(seed: int = _SEED + 2921) -> dict[str, float]:
    return _floats(_finite_blob("drinfeld_quotient", bench_drinfeld_quotient(seed)))


def bench_dg_nerve_family(seed: int = _SEED + 2922) -> dict[str, float]:
    return _floats(_finite_blob("dg_nerve", bench_dg_nerve(seed)))


def bench_keller_dg_family(seed: int = _SEED + 2923) -> dict[str, float]:
    return _floats(_finite_blob("keller_dg", bench_keller_dg(seed)))
