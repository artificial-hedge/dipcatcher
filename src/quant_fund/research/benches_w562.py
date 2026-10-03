"""Wave-562 symplectic-geometry-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ekeland_hofer import bench_ekeland_hofer
from quant_fund.models.gromov_width import bench_gromov_width
from quant_fund.models.hofer_metric import bench_hofer_metric
from quant_fund.models.mcduff_polterovich import (
    bench_mcduff_polterovich,
)
from quant_fund.models.symplectic_capacity import (
    bench_symplectic_capacity,
)
from quant_fund.models.symplectic_packing import (
    bench_symplectic_packing,
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


def bench_gromov_width_family(seed: int = _SEED + 3278) -> dict[str, float]:
    return _floats(_finite_blob("gromov_width", bench_gromov_width(seed)))


def bench_hofer_metric_family(seed: int = _SEED + 3279) -> dict[str, float]:
    return _floats(_finite_blob("hofer_metric", bench_hofer_metric(seed)))


def bench_symplectic_capacity_family(
    seed: int = _SEED + 3280,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "symplectic_capacity",
            bench_symplectic_capacity(seed),
        )
    )


def bench_symplectic_packing_family(
    seed: int = _SEED + 3281,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "symplectic_packing",
            bench_symplectic_packing(seed),
        )
    )


def bench_mcduff_polterovich_family(
    seed: int = _SEED + 3282,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mcduff_polterovich",
            bench_mcduff_polterovich(seed),
        )
    )


def bench_ekeland_hofer_family(seed: int = _SEED + 3283) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ekeland_hofer",
            bench_ekeland_hofer(seed),
        )
    )
