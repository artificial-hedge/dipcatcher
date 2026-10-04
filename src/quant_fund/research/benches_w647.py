"""Wave-647 homotopy-19 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.arkowitz_htpy import (
    bench_arkowitz_htpy,
)
from quant_fund.models.bochner_htpy import bench_bochner_htpy
from quant_fund.models.kahn_priddy import bench_kahn_priddy
from quant_fund.models.lin_htpy import bench_lin_htpy
from quant_fund.models.selick_htpy import bench_selick_htpy
from quant_fund.models.tits_building import (
    bench_tits_building,
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


def bench_selick_htpy_family(
    seed: int = _SEED + 3788,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "selick_htpy",
            bench_selick_htpy(seed),
        )
    )


def bench_arkowitz_htpy_family(
    seed: int = _SEED + 3789,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "arkowitz_htpy",
            bench_arkowitz_htpy(seed),
        )
    )


def bench_lin_htpy_family(
    seed: int = _SEED + 3790,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "lin_htpy",
            bench_lin_htpy(seed),
        )
    )


def bench_kahn_priddy_family(
    seed: int = _SEED + 3791,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kahn_priddy",
            bench_kahn_priddy(seed),
        )
    )


def bench_bochner_htpy_family(
    seed: int = _SEED + 3792,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bochner_htpy",
            bench_bochner_htpy(seed),
        )
    )


def bench_tits_building_family(
    seed: int = _SEED + 3793,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tits_building",
            bench_tits_building(seed),
        )
    )
