"""Wave-518 Yang-Baxter/braid bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.braid_rep import bench_braid_rep
from quant_fund.models.quantum_double import bench_quantum_double
from quant_fund.models.ribbon_cat import bench_ribbon_cat
from quant_fund.models.rtt_formalism import bench_rtt_formalism
from quant_fund.models.yang_baxter import bench_yang_baxter
from quant_fund.models.yangian import bench_yangian

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


def bench_yang_baxter_family(seed: int = _SEED + 3014) -> dict[str, float]:
    return _floats(_finite_blob("yang_baxter", bench_yang_baxter(seed)))


def bench_braid_rep_family(seed: int = _SEED + 3015) -> dict[str, float]:
    return _floats(_finite_blob("braid_rep", bench_braid_rep(seed)))


def bench_yangian_family(seed: int = _SEED + 3016) -> dict[str, float]:
    return _floats(_finite_blob("yangian", bench_yangian(seed)))


def bench_rtt_formalism_family(seed: int = _SEED + 3017) -> dict[str, float]:
    return _floats(_finite_blob("rtt_formalism", bench_rtt_formalism(seed)))


def bench_quantum_double_family(seed: int = _SEED + 3018) -> dict[str, float]:
    return _floats(_finite_blob("quantum_double", bench_quantum_double(seed)))


def bench_ribbon_cat_family(seed: int = _SEED + 3019) -> dict[str, float]:
    return _floats(_finite_blob("ribbon_cat", bench_ribbon_cat(seed)))
