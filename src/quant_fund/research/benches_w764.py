"""Wave-764 empirical-process bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.donsker_class import bench_donsker_class
from quant_fund.models.donsker_thm import bench_donsker_thm
from quant_fund.models.dz_invariance import bench_dz_invariance
from quant_fund.models.empirical_process import (
    bench_empirical_process,
)
from quant_fund.models.osj_metric import bench_osj_metric
from quant_fund.models.wiener_measure import bench_wiener_measure

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


def bench_wiener_measure_family(
    seed: int = _SEED + 15300,
) -> dict[str, float]:
    return _floats(_finite_blob("wiener_measure", bench_wiener_measure(seed)))


def bench_dz_invariance_family(
    seed: int = _SEED + 15301,
) -> dict[str, float]:
    return _floats(_finite_blob("dz_invariance", bench_dz_invariance(seed)))


def bench_donsker_thm_family(
    seed: int = _SEED + 15302,
) -> dict[str, float]:
    return _floats(_finite_blob("donsker_thm", bench_donsker_thm(seed)))


def bench_empirical_process_family(
    seed: int = _SEED + 15303,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "empirical_process",
            bench_empirical_process(seed),
        )
    )


def bench_donsker_class_family(
    seed: int = _SEED + 15304,
) -> dict[str, float]:
    return _floats(_finite_blob("donsker_class", bench_donsker_class(seed)))


def bench_osj_metric_family(
    seed: int = _SEED + 15305,
) -> dict[str, float]:
    return _floats(_finite_blob("osj_metric", bench_osj_metric(seed)))
