"""Wave-784 Levy bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.girsanov_thm2 import (
    bench_girsanov_thm2,
)
from quant_fund.models.levy_khinchine import (
    bench_levy_khinchine,
)
from quant_fund.models.levy_measure import (
    bench_levy_measure,
)
from quant_fund.models.self_decomp import bench_self_decomp
from quant_fund.models.stable_levy import bench_stable_levy
from quant_fund.models.subordinator import (
    bench_subordinator,
)

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


def bench_levy_khinchine_family(
    seed: int = _SEED + 17300,
) -> dict[str, float]:
    return _floats(_finite_blob("levy_khinchine", bench_levy_khinchine(seed)))


def bench_subordinator_family(
    seed: int = _SEED + 17301,
) -> dict[str, float]:
    return _floats(_finite_blob("subordinator", bench_subordinator(seed)))


def bench_stable_levy_family(
    seed: int = _SEED + 17302,
) -> dict[str, float]:
    return _floats(_finite_blob("stable_levy", bench_stable_levy(seed)))


def bench_self_decomp_family(
    seed: int = _SEED + 17303,
) -> dict[str, float]:
    return _floats(_finite_blob("self_decomp", bench_self_decomp(seed)))


def bench_levy_measure_family(
    seed: int = _SEED + 17304,
) -> dict[str, float]:
    return _floats(_finite_blob("levy_measure", bench_levy_measure(seed)))


def bench_girsanov_thm2_family(
    seed: int = _SEED + 17305,
) -> dict[str, float]:
    return _floats(_finite_blob("girsanov_thm2", bench_girsanov_thm2(seed)))
