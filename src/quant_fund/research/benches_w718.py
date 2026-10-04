"""Wave-718 representation-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.der_bimodule import bench_der_bimodule
from quant_fund.models.helix_theory import bench_helix_theory
from quant_fund.models.higher_auslander import (
    bench_higher_auslander,
)
from quant_fund.models.icy_paper import bench_icy_paper
from quant_fund.models.mutation_class import (
    bench_mutation_class,
)
from quant_fund.models.rep_finite import bench_rep_finite

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


def bench_helix_theory_family(
    seed: int = _SEED + 10700,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "helix_theory",
            bench_helix_theory(seed),
        )
    )


def bench_mutation_class_family(
    seed: int = _SEED + 10701,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mutation_class",
            bench_mutation_class(seed),
        )
    )


def bench_rep_finite_family(
    seed: int = _SEED + 10702,
) -> dict[str, float]:
    return _floats(_finite_blob("rep_finite", bench_rep_finite(seed)))


def bench_der_bimodule_family(
    seed: int = _SEED + 10703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "der_bimodule",
            bench_der_bimodule(seed),
        )
    )


def bench_icy_paper_family(
    seed: int = _SEED + 10704,
) -> dict[str, float]:
    return _floats(_finite_blob("icy_paper", bench_icy_paper(seed)))


def bench_higher_auslander_family(
    seed: int = _SEED + 10705,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "higher_auslander",
            bench_higher_auslander(seed),
        )
    )
