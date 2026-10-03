"""Wave-715 cluster-algebra bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.auslander_reiten import (
    bench_auslander_reiten,
)
from quant_fund.models.cluster_algebra import (
    bench_cluster_algebra,
)
from quant_fund.models.cluster_category import (
    bench_cluster_category,
)
from quant_fund.models.quiver_mutation import (
    bench_quiver_mutation,
)
from quant_fund.models.silting_object import (
    bench_silting_object,
)
from quant_fund.models.tilting_object import (
    bench_tilting_object,
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


def bench_cluster_algebra_family(
    seed: int = _SEED + 10400,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cluster_algebra",
            bench_cluster_algebra(seed),
        )
    )


def bench_quiver_mutation_family(
    seed: int = _SEED + 10401,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "quiver_mutation",
            bench_quiver_mutation(seed),
        )
    )


def bench_tilting_object_family(
    seed: int = _SEED + 10402,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tilting_object",
            bench_tilting_object(seed),
        )
    )


def bench_auslander_reiten_family(
    seed: int = _SEED + 10403,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "auslander_reiten",
            bench_auslander_reiten(seed),
        )
    )


def bench_cluster_category_family(
    seed: int = _SEED + 10404,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cluster_category",
            bench_cluster_category(seed),
        )
    )


def bench_silting_object_family(
    seed: int = _SEED + 10405,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "silting_object",
            bench_silting_object(seed),
        )
    )
