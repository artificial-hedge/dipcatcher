"""Wave-712 motivic-24 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.motivic_additive import (
    bench_motivic_additive,
)
from quant_fund.models.motivic_additive_cat import (
    bench_motivic_additive_cat,
)
from quant_fund.models.motivic_chern2 import (
    bench_motivic_chern2,
)
from quant_fund.models.motivic_cover import (
    bench_motivic_cover,
)
from quant_fund.models.motivic_filtration2 import (
    bench_motivic_filtration2,
)
from quant_fund.models.motivic_gysin2 import (
    bench_motivic_gysin2,
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


def bench_motivic_additive_family(
    seed: int = _SEED + 10100,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_additive",
            bench_motivic_additive(seed),
        )
    )


def bench_motivic_additive_cat_family(
    seed: int = _SEED + 10101,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_additive_cat",
            bench_motivic_additive_cat(seed),
        )
    )


def bench_motivic_cover_family(
    seed: int = _SEED + 10102,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_cover",
            bench_motivic_cover(seed),
        )
    )


def bench_motivic_gysin2_family(
    seed: int = _SEED + 10103,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_gysin2",
            bench_motivic_gysin2(seed),
        )
    )


def bench_motivic_chern2_family(
    seed: int = _SEED + 10104,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_chern2",
            bench_motivic_chern2(seed),
        )
    )


def bench_motivic_filtration2_family(
    seed: int = _SEED + 10105,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_filtration2",
            bench_motivic_filtration2(seed),
        )
    )
