"""Wave-559 complex-geometry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.calabi_conjecture import bench_calabi_conjecture
from quant_fund.models.calabi_yau_mfd import bench_calabi_yau_mfd
from quant_fund.models.csck_metric import bench_csck_metric
from quant_fund.models.futaki_invariant import bench_futaki_invariant
from quant_fund.models.k_stability import bench_k_stability
from quant_fund.models.kahler_einstein import bench_kahler_einstein

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


def bench_calabi_yau_mfd_family(seed: int = _SEED + 3260) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "calabi_yau_mfd",
            bench_calabi_yau_mfd(seed),
        )
    )


def bench_calabi_conjecture_family(
    seed: int = _SEED + 3261,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "calabi_conjecture",
            bench_calabi_conjecture(seed),
        )
    )


def bench_kahler_einstein_family(
    seed: int = _SEED + 3262,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kahler_einstein",
            bench_kahler_einstein(seed),
        )
    )


def bench_k_stability_family(seed: int = _SEED + 3263) -> dict[str, float]:
    return _floats(_finite_blob("k_stability", bench_k_stability(seed)))


def bench_csck_metric_family(seed: int = _SEED + 3264) -> dict[str, float]:
    return _floats(_finite_blob("csck_metric", bench_csck_metric(seed)))


def bench_futaki_invariant_family(
    seed: int = _SEED + 3265,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "futaki_invariant",
            bench_futaki_invariant(seed),
        )
    )
