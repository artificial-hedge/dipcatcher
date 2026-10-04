"""Wave-824 stochastic-order bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.convex_order import (
    bench_convex_order,
)
from quant_fund.models.first_order_dom import (
    bench_first_order_dom,
)
from quant_fund.models.hazard_rate_order import (
    bench_hazard_rate_order,
)
from quant_fund.models.second_order_dom import (
    bench_second_order_dom,
)
from quant_fund.models.supermodular_order import (
    bench_supermodular_order,
)
from quant_fund.models.usual_stoch_order import (
    bench_usual_stoch_order,
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


def bench_usual_stoch_order_family(
    seed: int = _SEED + 21200,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "usual_stoch_order",
            bench_usual_stoch_order(seed),
        )
    )


def bench_first_order_dom_family(
    seed: int = _SEED + 21201,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "first_order_dom",
            bench_first_order_dom(seed),
        )
    )


def bench_second_order_dom_family(
    seed: int = _SEED + 21202,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "second_order_dom",
            bench_second_order_dom(seed),
        )
    )


def bench_convex_order_family(
    seed: int = _SEED + 21203,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "convex_order",
            bench_convex_order(seed),
        )
    )


def bench_hazard_rate_order_family(
    seed: int = _SEED + 21204,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hazard_rate_order",
            bench_hazard_rate_order(seed),
        )
    )


def bench_supermodular_order_family(
    seed: int = _SEED + 21205,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "supermodular_order",
            bench_supermodular_order(seed),
        )
    )
