"""Wave-734 SLE bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.garmadon_sle import bench_garmadon_sle
from quant_fund.models.lawler_werner import bench_lawler_werner
from quant_fund.models.miller_sheffield import (
    bench_miller_sheffield,
)
from quant_fund.models.osgood_schramm import (
    bench_osgood_schramm,
)
from quant_fund.models.smirnov_parafermion import (
    bench_smirnov_parafermion,
)
from quant_fund.models.werner_wilson import bench_werner_wilson

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


def bench_osgood_schramm_family(
    seed: int = _SEED + 12300,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "osgood_schramm",
            bench_osgood_schramm(seed),
        )
    )


def bench_lawler_werner_family(
    seed: int = _SEED + 12301,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "lawler_werner",
            bench_lawler_werner(seed),
        )
    )


def bench_werner_wilson_family(
    seed: int = _SEED + 12302,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "werner_wilson",
            bench_werner_wilson(seed),
        )
    )


def bench_smirnov_parafermion_family(
    seed: int = _SEED + 12303,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "smirnov_parafermion",
            bench_smirnov_parafermion(seed),
        )
    )


def bench_garmadon_sle_family(
    seed: int = _SEED + 12304,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "garmadon_sle",
            bench_garmadon_sle(seed),
        )
    )


def bench_miller_sheffield_family(
    seed: int = _SEED + 12305,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "miller_sheffield",
            bench_miller_sheffield(seed),
        )
    )
