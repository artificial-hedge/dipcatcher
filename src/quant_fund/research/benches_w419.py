"""Wave-419 algebraic-topology-5 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bundle_section import bench_bundle_section
from quant_fund.models.classify_space import bench_classify_space
from quant_fund.models.path_fibration import bench_path_fibration
from quant_fund.models.serre_fibration import bench_serre_fibration
from quant_fund.models.thom_space import bench_thom_space
from quant_fund.models.vector_bundle import bench_vector_bundle

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


def bench_serre_fibration_family(
    seed: int = _SEED + 2420,
) -> dict[str, float]:
    return _floats(_finite_blob("serre_fibration", bench_serre_fibration(seed)))


def bench_path_fibration_family(
    seed: int = _SEED + 2421,
) -> dict[str, float]:
    return _floats(_finite_blob("path_fibration", bench_path_fibration(seed)))


def bench_bundle_section_family(
    seed: int = _SEED + 2422,
) -> dict[str, float]:
    return _floats(_finite_blob("bundle_section", bench_bundle_section(seed)))


def bench_classify_space_family(
    seed: int = _SEED + 2423,
) -> dict[str, float]:
    return _floats(_finite_blob("classify_space", bench_classify_space(seed)))


def bench_vector_bundle_family(
    seed: int = _SEED + 2424,
) -> dict[str, float]:
    return _floats(_finite_blob("vector_bundle", bench_vector_bundle(seed)))


def bench_thom_space_family(
    seed: int = _SEED + 2425,
) -> dict[str, float]:
    return _floats(_finite_blob("thom_space", bench_thom_space(seed)))
