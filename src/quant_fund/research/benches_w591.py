"""Wave-591 motivic-7 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.friedlander_voev import (
    bench_friedlander_voev,
)
from quant_fund.models.motivic_descent import (
    bench_motivic_descent,
)
from quant_fund.models.motivic_eilenberg import (
    bench_motivic_eilenberg,
)
from quant_fund.models.motivic_invert import (
    bench_motivic_invert,
)
from quant_fund.models.motivic_purity import (
    bench_motivic_purity,
)
from quant_fund.models.motivic_zeta import bench_motivic_zeta

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


def bench_friedlander_voev_family(
    seed: int = _SEED + 3452,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "friedlander_voev",
            bench_friedlander_voev(seed),
        )
    )


def bench_motivic_eilenberg_family(
    seed: int = _SEED + 3453,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_eilenberg",
            bench_motivic_eilenberg(seed),
        )
    )


def bench_motivic_zeta_family(
    seed: int = _SEED + 3454,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_zeta", bench_motivic_zeta(seed)))


def bench_motivic_purity_family(
    seed: int = _SEED + 3455,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_purity",
            bench_motivic_purity(seed),
        )
    )


def bench_motivic_descent_family(
    seed: int = _SEED + 3456,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_descent",
            bench_motivic_descent(seed),
        )
    )


def bench_motivic_invert_family(
    seed: int = _SEED + 3457,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_invert",
            bench_motivic_invert(seed),
        )
    )
