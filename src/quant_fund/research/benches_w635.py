"""Wave-635 commutative-algebra-5 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.excellent_ring import (
    bench_excellent_ring,
)
from quant_fund.models.going_up import bench_going_up
from quant_fund.models.integral_closure2 import (
    bench_integral_closure2,
)
from quant_fund.models.lying_over import bench_lying_over
from quant_fund.models.weil_divisor2 import (
    bench_weil_divisor2,
)
from quant_fund.models.zariski_main import (
    bench_zariski_main,
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


def bench_excellent_ring_family(
    seed: int = _SEED + 3716,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "excellent_ring",
            bench_excellent_ring(seed),
        )
    )


def bench_zariski_main_family(
    seed: int = _SEED + 3717,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "zariski_main",
            bench_zariski_main(seed),
        )
    )


def bench_going_up_family(
    seed: int = _SEED + 3718,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "going_up",
            bench_going_up(seed),
        )
    )


def bench_lying_over_family(
    seed: int = _SEED + 3719,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "lying_over",
            bench_lying_over(seed),
        )
    )


def bench_integral_closure2_family(
    seed: int = _SEED + 3720,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "integral_closure2",
            bench_integral_closure2(seed),
        )
    )


def bench_weil_divisor2_family(
    seed: int = _SEED + 3721,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "weil_divisor2",
            bench_weil_divisor2(seed),
        )
    )
