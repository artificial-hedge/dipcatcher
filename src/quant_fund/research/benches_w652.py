"""Wave-652 motivic-12 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.asymptotic_motive import bench_asymptotic_motive
from quant_fund.models.exponential_motive import bench_exponential_motive
from quant_fund.models.log_motive import bench_log_motive
from quant_fund.models.numerical_motive import bench_numerical_motive
from quant_fund.models.sheaf_motive import bench_sheaf_motive
from quant_fund.models.strict_motive import bench_strict_motive

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


def bench_strict_motive_family(
    seed: int = _SEED + 4100,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "strict_motive",
            bench_strict_motive(seed),
        )
    )


def bench_sheaf_motive_family(
    seed: int = _SEED + 4101,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "sheaf_motive",
            bench_sheaf_motive(seed),
        )
    )


def bench_numerical_motive_family(
    seed: int = _SEED + 4102,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "numerical_motive",
            bench_numerical_motive(seed),
        )
    )


def bench_asymptotic_motive_family(
    seed: int = _SEED + 4103,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "asymptotic_motive",
            bench_asymptotic_motive(seed),
        )
    )


def bench_exponential_motive_family(
    seed: int = _SEED + 4104,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "exponential_motive",
            bench_exponential_motive(seed),
        )
    )


def bench_log_motive_family(
    seed: int = _SEED + 4105,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "log_motive",
            bench_log_motive(seed),
        )
    )
