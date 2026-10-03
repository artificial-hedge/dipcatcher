"""Wave-570 positivity/moduli bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bogomolov_ineq import bench_bogomolov_ineq
from quant_fund.models.boundedness_moduli import (
    bench_boundedness_moduli,
)
from quant_fund.models.hodge_index import bench_hodge_index
from quant_fund.models.kodaira_vanishing import (
    bench_kodaira_vanishing,
)
from quant_fund.models.kollar_mori import bench_kollar_mori
from quant_fund.models.stability_sheaf import bench_stability_sheaf

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


def bench_hodge_index_family(seed: int = _SEED + 3326) -> dict[str, float]:
    return _floats(_finite_blob("hodge_index", bench_hodge_index(seed)))


def bench_kodaira_vanishing_family(
    seed: int = _SEED + 3327,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kodaira_vanishing",
            bench_kodaira_vanishing(seed),
        )
    )


def bench_kollar_mori_family(seed: int = _SEED + 3328) -> dict[str, float]:
    return _floats(_finite_blob("kollar_mori", bench_kollar_mori(seed)))


def bench_boundedness_moduli_family(
    seed: int = _SEED + 3329,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "boundedness_moduli",
            bench_boundedness_moduli(seed),
        )
    )


def bench_stability_sheaf_family(
    seed: int = _SEED + 3330,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stability_sheaf",
            bench_stability_sheaf(seed),
        )
    )


def bench_bogomolov_ineq_family(
    seed: int = _SEED + 3331,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bogomolov_ineq",
            bench_bogomolov_ineq(seed),
        )
    )
