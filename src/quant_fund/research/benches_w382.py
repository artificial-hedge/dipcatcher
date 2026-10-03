"""Wave-382 model-theory-4 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.back_forth import bench_back_forth
from quant_fund.models.indiscernibles import bench_indiscernibles
from quant_fund.models.omitting_types import bench_omitting_types
from quant_fund.models.saturation_test import bench_saturation_test
from quant_fund.models.stability_spec import bench_stability_spec
from quant_fund.models.stone_duality import bench_stone_duality

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


def bench_stone_duality_family(seed: int = _SEED + 2198) -> dict[str, float]:
    return _floats(_finite_blob("stone_duality", bench_stone_duality(seed)))


def bench_saturation_test_family(seed: int = _SEED + 2199) -> dict[str, float]:
    return _floats(_finite_blob("saturation_test", bench_saturation_test(seed)))


def bench_omitting_types_family(seed: int = _SEED + 2200) -> dict[str, float]:
    return _floats(_finite_blob("omitting_types", bench_omitting_types(seed)))


def bench_indiscernibles_family(seed: int = _SEED + 2201) -> dict[str, float]:
    return _floats(_finite_blob("indiscernibles", bench_indiscernibles(seed)))


def bench_stability_spec_family(seed: int = _SEED + 2202) -> dict[str, float]:
    return _floats(_finite_blob("stability_spec", bench_stability_spec(seed)))


def bench_back_forth_family(seed: int = _SEED + 2203) -> dict[str, float]:
    return _floats(_finite_blob("back_forth", bench_back_forth(seed)))
