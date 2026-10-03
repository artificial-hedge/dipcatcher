"""Wave-571 singularity-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bernstein_sato import bench_bernstein_sato
from quant_fund.models.du_val_sing import bench_du_val_sing
from quant_fund.models.log_canonical import bench_log_canonical
from quant_fund.models.milnor_fiber import bench_milnor_fiber
from quant_fund.models.multiplier_ideal import (
    bench_multiplier_ideal,
)
from quant_fund.models.rational_sing import bench_rational_sing

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


def bench_du_val_sing_family(seed: int = _SEED + 3332) -> dict[str, float]:
    return _floats(_finite_blob("du_val_sing", bench_du_val_sing(seed)))


def bench_rational_sing_family(
    seed: int = _SEED + 3333,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "rational_sing",
            bench_rational_sing(seed),
        )
    )


def bench_log_canonical_family(
    seed: int = _SEED + 3334,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "log_canonical",
            bench_log_canonical(seed),
        )
    )


def bench_multiplier_ideal_family(
    seed: int = _SEED + 3335,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "multiplier_ideal",
            bench_multiplier_ideal(seed),
        )
    )


def bench_bernstein_sato_family(
    seed: int = _SEED + 3336,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bernstein_sato",
            bench_bernstein_sato(seed),
        )
    )


def bench_milnor_fiber_family(seed: int = _SEED + 3337) -> dict[str, float]:
    return _floats(_finite_blob("milnor_fiber", bench_milnor_fiber(seed)))
