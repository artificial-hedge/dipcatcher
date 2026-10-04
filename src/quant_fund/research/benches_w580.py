"""Wave-580 enumerative-combinatorics bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cycle_index import bench_cycle_index
from quant_fund.models.exponential_gf import (
    bench_exponential_gf,
)
from quant_fund.models.lagrange_inversion import (
    bench_lagrange_inversion,
)
from quant_fund.models.matrix_tree import bench_matrix_tree
from quant_fund.models.species_theory import (
    bench_species_theory,
)
from quant_fund.models.transfer_matrix import (
    bench_transfer_matrix,
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


def bench_species_theory_family(
    seed: int = _SEED + 3386,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "species_theory",
            bench_species_theory(seed),
        )
    )


def bench_cycle_index_family(seed: int = _SEED + 3387) -> dict[str, float]:
    return _floats(_finite_blob("cycle_index", bench_cycle_index(seed)))


def bench_lagrange_inversion_family(
    seed: int = _SEED + 3388,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "lagrange_inversion",
            bench_lagrange_inversion(seed),
        )
    )


def bench_transfer_matrix_family(
    seed: int = _SEED + 3389,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "transfer_matrix",
            bench_transfer_matrix(seed),
        )
    )


def bench_matrix_tree_family(seed: int = _SEED + 3390) -> dict[str, float]:
    return _floats(_finite_blob("matrix_tree", bench_matrix_tree(seed)))


def bench_exponential_gf_family(
    seed: int = _SEED + 3391,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "exponential_gf",
            bench_exponential_gf(seed),
        )
    )
