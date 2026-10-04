"""Wave-714 motivic-25 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.derived_proper2 import (
    bench_derived_proper2,
)
from quant_fund.models.derived_separated2 import (
    bench_derived_separated2,
)
from quant_fund.models.motivic_functor import (
    bench_motivic_functor,
)
from quant_fund.models.motivic_nerve import bench_motivic_nerve
from quant_fund.models.motivic_partial import (
    bench_motivic_partial,
)
from quant_fund.models.motivic_total import bench_motivic_total

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


def bench_motivic_total_family(
    seed: int = _SEED + 10300,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_total", bench_motivic_total(seed)))


def bench_motivic_partial_family(
    seed: int = _SEED + 10301,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_partial",
            bench_motivic_partial(seed),
        )
    )


def bench_motivic_functor_family(
    seed: int = _SEED + 10302,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_functor",
            bench_motivic_functor(seed),
        )
    )


def bench_motivic_nerve_family(
    seed: int = _SEED + 10303,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_nerve", bench_motivic_nerve(seed)))


def bench_derived_proper2_family(
    seed: int = _SEED + 10304,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "derived_proper2",
            bench_derived_proper2(seed),
        )
    )


def bench_derived_separated2_family(
    seed: int = _SEED + 10305,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "derived_separated2",
            bench_derived_separated2(seed),
        )
    )
