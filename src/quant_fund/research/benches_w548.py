"""Wave-548 Floer-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.floer_homology import bench_floer_homology
from quant_fund.models.fukaya_cat import bench_fukaya_cat
from quant_fund.models.instanton_floer import bench_instanton_floer
from quant_fund.models.knot_floer import bench_knot_floer
from quant_fund.models.lagrangian_floer import bench_lagrangian_floer
from quant_fund.models.monopole_floer import bench_monopole_floer

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


def bench_floer_homology_family(seed: int = _SEED + 3194) -> dict[str, float]:
    return _floats(_finite_blob("floer_homology", bench_floer_homology(seed)))


def bench_knot_floer_family(seed: int = _SEED + 3195) -> dict[str, float]:
    return _floats(_finite_blob("knot_floer", bench_knot_floer(seed)))


def bench_instanton_floer_family(
    seed: int = _SEED + 3196,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "instanton_floer",
            bench_instanton_floer(seed),
        )
    )


def bench_monopole_floer_family(
    seed: int = _SEED + 3197,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "monopole_floer",
            bench_monopole_floer(seed),
        )
    )


def bench_lagrangian_floer_family(
    seed: int = _SEED + 3198,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "lagrangian_floer",
            bench_lagrangian_floer(seed),
        )
    )


def bench_fukaya_cat_family(seed: int = _SEED + 3199) -> dict[str, float]:
    return _floats(_finite_blob("fukaya_cat", bench_fukaya_cat(seed)))
