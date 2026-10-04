"""Wave-819 stopping-time bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.accessible_time import (
    bench_accessible_time,
)
from quant_fund.models.debuts_theorem import (
    bench_debuts_theorem,
)
from quant_fund.models.first_hitting import (
    bench_first_hitting,
)
from quant_fund.models.last_exit import (
    bench_last_exit,
)
from quant_fund.models.progressive_set import (
    bench_progressive_set,
)
from quant_fund.models.stopping_sigma import (
    bench_stopping_sigma,
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


def bench_first_hitting_family(
    seed: int = _SEED + 20700,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "first_hitting",
            bench_first_hitting(seed),
        )
    )


def bench_last_exit_family(
    seed: int = _SEED + 20701,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "last_exit",
            bench_last_exit(seed),
        )
    )


def bench_stopping_sigma_family(
    seed: int = _SEED + 20702,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stopping_sigma",
            bench_stopping_sigma(seed),
        )
    )


def bench_progressive_set_family(
    seed: int = _SEED + 20703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "progressive_set",
            bench_progressive_set(seed),
        )
    )


def bench_debuts_theorem_family(
    seed: int = _SEED + 20704,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "debuts_theorem",
            bench_debuts_theorem(seed),
        )
    )


def bench_accessible_time_family(
    seed: int = _SEED + 20705,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "accessible_time",
            bench_accessible_time(seed),
        )
    )
