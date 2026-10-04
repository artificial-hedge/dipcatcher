"""Wave-690 homotopy-27 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.homotopy_model import (
    bench_homotopy_model,
)
from quant_fund.models.homotopy_sheaf import (
    bench_homotopy_sheaf,
)
from quant_fund.models.stable_algebra import (
    bench_stable_algebra,
)
from quant_fund.models.stable_group import bench_stable_group
from quant_fund.models.stable_module import (
    bench_stable_module,
)
from quant_fund.models.stable_monoid import (
    bench_stable_monoid,
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


def bench_homotopy_sheaf_family(
    seed: int = _SEED + 7900,
) -> dict[str, float]:
    return _floats(_finite_blob("homotopy_sheaf", bench_homotopy_sheaf(seed)))


def bench_homotopy_model_family(
    seed: int = _SEED + 7901,
) -> dict[str, float]:
    return _floats(_finite_blob("homotopy_model", bench_homotopy_model(seed)))


def bench_stable_monoid_family(
    seed: int = _SEED + 7902,
) -> dict[str, float]:
    return _floats(_finite_blob("stable_monoid", bench_stable_monoid(seed)))


def bench_stable_group_family(
    seed: int = _SEED + 7903,
) -> dict[str, float]:
    return _floats(_finite_blob("stable_group", bench_stable_group(seed)))


def bench_stable_module_family(
    seed: int = _SEED + 7904,
) -> dict[str, float]:
    return _floats(_finite_blob("stable_module", bench_stable_module(seed)))


def bench_stable_algebra_family(
    seed: int = _SEED + 7905,
) -> dict[str, float]:
    return _floats(_finite_blob("stable_algebra", bench_stable_algebra(seed)))
