"""Wave-898 BVP bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.collocation_bvp import (
    bench_collocation_bvp,
)
from quant_fund.models.finite_diff_bvp import (
    bench_finite_diff_bvp,
)
from quant_fund.models.multiple_shooting import (
    bench_multiple_shooting,
)
from quant_fund.models.relaxation_bvp import (
    bench_relaxation_bvp,
)
from quant_fund.models.riccati_bvp import (
    bench_riccati_bvp,
)
from quant_fund.models.shooting_bvp import (
    bench_shooting_bvp,
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


def bench_shooting_bvp_family(
    seed: int = _SEED + 28600,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "shooting_bvp",
            bench_shooting_bvp(seed),
        )
    )


def bench_multiple_shooting_family(
    seed: int = _SEED + 28601,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "multiple_shooting",
            bench_multiple_shooting(seed),
        )
    )


def bench_collocation_bvp_family(
    seed: int = _SEED + 28602,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "collocation_bvp",
            bench_collocation_bvp(seed),
        )
    )


def bench_finite_diff_bvp_family(
    seed: int = _SEED + 28603,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "finite_diff_bvp",
            bench_finite_diff_bvp(seed),
        )
    )


def bench_relaxation_bvp_family(
    seed: int = _SEED + 28604,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "relaxation_bvp",
            bench_relaxation_bvp(seed),
        )
    )


def bench_riccati_bvp_family(
    seed: int = _SEED + 28605,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "riccati_bvp",
            bench_riccati_bvp(seed),
        )
    )
