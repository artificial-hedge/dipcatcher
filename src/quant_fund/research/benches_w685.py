"""Wave-685 homotopy-26 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.homotopy_class2 import bench_homotopy_class2
from quant_fund.models.homotopy_limit import bench_homotopy_limit
from quant_fund.models.homotopy_tower import bench_homotopy_tower
from quant_fund.models.spectral_sequence5 import (
    bench_spectral_sequence5,
)
from quant_fund.models.stable_bousfield import (
    bench_stable_bousfield,
)
from quant_fund.models.stable_mapping import bench_stable_mapping

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


def bench_homotopy_limit_family(
    seed: int = _SEED + 7400,
) -> dict[str, float]:
    return _floats(_finite_blob("homotopy_limit", bench_homotopy_limit(seed)))


def bench_homotopy_tower_family(
    seed: int = _SEED + 7401,
) -> dict[str, float]:
    return _floats(_finite_blob("homotopy_tower", bench_homotopy_tower(seed)))


def bench_spectral_sequence5_family(
    seed: int = _SEED + 7402,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_sequence5",
            bench_spectral_sequence5(seed),
        )
    )


def bench_homotopy_class2_family(
    seed: int = _SEED + 7403,
) -> dict[str, float]:
    return _floats(_finite_blob("homotopy_class2", bench_homotopy_class2(seed)))


def bench_stable_mapping_family(
    seed: int = _SEED + 7404,
) -> dict[str, float]:
    return _floats(_finite_blob("stable_mapping", bench_stable_mapping(seed)))


def bench_stable_bousfield_family(
    seed: int = _SEED + 7405,
) -> dict[str, float]:
    return _floats(_finite_blob("stable_bousfield", bench_stable_bousfield(seed)))
