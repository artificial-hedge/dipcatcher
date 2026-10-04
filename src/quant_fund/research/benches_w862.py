"""Wave-862 sparse-grid/dimension-adaptive bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.anisotropic_quad import (
    bench_anisotropic_quad,
)
from quant_fund.models.combination_technique import (
    bench_combination_technique,
)
from quant_fund.models.dimension_adaptive import (
    bench_dimension_adaptive,
)
from quant_fund.models.gerstner_griebel import (
    bench_gerstner_griebel,
)
from quant_fund.models.smolyak_grid import (
    bench_smolyak_grid,
)
from quant_fund.models.sparse_tensor import (
    bench_sparse_tensor,
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


def bench_smolyak_grid_family(
    seed: int = _SEED + 25000,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "smolyak_grid",
            bench_smolyak_grid(seed),
        )
    )


def bench_sparse_tensor_family(
    seed: int = _SEED + 25001,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "sparse_tensor",
            bench_sparse_tensor(seed),
        )
    )


def bench_anisotropic_quad_family(
    seed: int = _SEED + 25002,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "anisotropic_quad",
            bench_anisotropic_quad(seed),
        )
    )


def bench_gerstner_griebel_family(
    seed: int = _SEED + 25003,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gerstner_griebel",
            bench_gerstner_griebel(seed),
        )
    )


def bench_combination_technique_family(
    seed: int = _SEED + 25004,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "combination_technique",
            bench_combination_technique(seed),
        )
    )


def bench_dimension_adaptive_family(
    seed: int = _SEED + 25005,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dimension_adaptive",
            bench_dimension_adaptive(seed),
        )
    )
