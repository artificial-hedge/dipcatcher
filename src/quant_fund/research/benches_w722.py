"""Wave-722 special-values bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.borel_motivic import bench_borel_motivic
from quant_fund.models.deligne_period import bench_deligne_period
from quant_fund.models.motivic_multiple_zeta import (
    bench_motivic_multiple_zeta,
)
from quant_fund.models.period_poly import bench_period_poly
from quant_fund.models.specialization_motive import (
    bench_specialization_motive,
)
from quant_fund.models.zagier_polylog import bench_zagier_polylog

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


def bench_period_poly_family(
    seed: int = _SEED + 11100,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "period_poly",
            bench_period_poly(seed),
        )
    )


def bench_specialization_motive_family(
    seed: int = _SEED + 11101,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "specialization_motive",
            bench_specialization_motive(seed),
        )
    )


def bench_borel_motivic_family(
    seed: int = _SEED + 11102,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "borel_motivic",
            bench_borel_motivic(seed),
        )
    )


def bench_zagier_polylog_family(
    seed: int = _SEED + 11103,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "zagier_polylog",
            bench_zagier_polylog(seed),
        )
    )


def bench_deligne_period_family(
    seed: int = _SEED + 11104,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "deligne_period",
            bench_deligne_period(seed),
        )
    )


def bench_motivic_multiple_zeta_family(
    seed: int = _SEED + 11105,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_multiple_zeta",
            bench_motivic_multiple_zeta(seed),
        )
    )
