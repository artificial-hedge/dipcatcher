"""Wave-783 stochastic-order bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cramer_wold import bench_cramer_wold
from quant_fund.models.hazard_order import bench_hazard_order
from quant_fund.models.likelihood_order import (
    bench_likelihood_order,
)
from quant_fund.models.predictable_bracket import (
    bench_predictable_bracket,
)
from quant_fund.models.semi_mart import bench_semi_mart
from quant_fund.models.stricker_thm import bench_stricker_thm

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


def bench_semi_mart_family(
    seed: int = _SEED + 17200,
) -> dict[str, float]:
    return _floats(_finite_blob("semi_mart", bench_semi_mart(seed)))


def bench_predictable_bracket_family(
    seed: int = _SEED + 17201,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "predictable_bracket",
            bench_predictable_bracket(seed),
        )
    )


def bench_cramer_wold_family(
    seed: int = _SEED + 17202,
) -> dict[str, float]:
    return _floats(_finite_blob("cramer_wold", bench_cramer_wold(seed)))


def bench_stricker_thm_family(
    seed: int = _SEED + 17203,
) -> dict[str, float]:
    return _floats(_finite_blob("stricker_thm", bench_stricker_thm(seed)))


def bench_likelihood_order_family(
    seed: int = _SEED + 17204,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "likelihood_order",
            bench_likelihood_order(seed),
        )
    )


def bench_hazard_order_family(
    seed: int = _SEED + 17205,
) -> dict[str, float]:
    return _floats(_finite_blob("hazard_order", bench_hazard_order(seed)))
