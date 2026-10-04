"""Wave-603 topos-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.atomic_topos import bench_atomic_topos
from quant_fund.models.classifying_topos import (
    bench_classifying_topos,
)
from quant_fund.models.essential_morph import (
    bench_essential_morph,
)
from quant_fund.models.giraud_axiom import bench_giraud_axiom
from quant_fund.models.logical_morph import bench_logical_morph
from quant_fund.models.slice_topos import bench_slice_topos

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


def bench_slice_topos_family(
    seed: int = _SEED + 3524,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "slice_topos",
            bench_slice_topos(seed),
        )
    )


def bench_logical_morph_family(
    seed: int = _SEED + 3525,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "logical_morph",
            bench_logical_morph(seed),
        )
    )


def bench_classifying_topos_family(
    seed: int = _SEED + 3526,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "classifying_topos",
            bench_classifying_topos(seed),
        )
    )


def bench_atomic_topos_family(
    seed: int = _SEED + 3527,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "atomic_topos",
            bench_atomic_topos(seed),
        )
    )


def bench_essential_morph_family(
    seed: int = _SEED + 3528,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "essential_morph",
            bench_essential_morph(seed),
        )
    )


def bench_giraud_axiom_family(
    seed: int = _SEED + 3529,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "giraud_axiom",
            bench_giraud_axiom(seed),
        )
    )
