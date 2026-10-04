"""Wave-720 derived-dimension bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.categorical_entropy import (
    bench_categorical_entropy,
)
from quant_fund.models.cluster_tilting import (
    bench_cluster_tilting,
)
from quant_fund.models.derived_morita import (
    bench_derived_morita,
)
from quant_fund.models.preprojective_alg import (
    bench_preprojective_alg,
)
from quant_fund.models.rouquier_dim import bench_rouquier_dim
from quant_fund.models.serre_dim import bench_serre_dim

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


def bench_cluster_tilting_family(
    seed: int = _SEED + 10900,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cluster_tilting",
            bench_cluster_tilting(seed),
        )
    )


def bench_derived_morita_family(
    seed: int = _SEED + 10901,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "derived_morita",
            bench_derived_morita(seed),
        )
    )


def bench_preprojective_alg_family(
    seed: int = _SEED + 10902,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "preprojective_alg",
            bench_preprojective_alg(seed),
        )
    )


def bench_categorical_entropy_family(
    seed: int = _SEED + 10903,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "categorical_entropy",
            bench_categorical_entropy(seed),
        )
    )


def bench_serre_dim_family(
    seed: int = _SEED + 10904,
) -> dict[str, float]:
    return _floats(_finite_blob("serre_dim", bench_serre_dim(seed)))


def bench_rouquier_dim_family(
    seed: int = _SEED + 10905,
) -> dict[str, float]:
    return _floats(_finite_blob("rouquier_dim", bench_rouquier_dim(seed)))
