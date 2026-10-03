"""Wave-728 motivic-A1 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.a1_degrees import bench_a1_degrees
from quant_fund.models.emerton_glass import (
    bench_emerton_glass,
)
from quant_fund.models.luan_yao import bench_luan_yao
from quant_fund.models.morel_voev import bench_morel_voev
from quant_fund.models.totaro_cycle import bench_totaro_cycle
from quant_fund.models.voev_homotopy import bench_voev_homotopy

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


def bench_emerton_glass_family(
    seed: int = _SEED + 11700,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "emerton_glass",
            bench_emerton_glass(seed),
        )
    )


def bench_luan_yao_family(
    seed: int = _SEED + 11701,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "luan_yao",
            bench_luan_yao(seed),
        )
    )


def bench_morel_voev_family(
    seed: int = _SEED + 11702,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "morel_voev",
            bench_morel_voev(seed),
        )
    )


def bench_voev_homotopy_family(
    seed: int = _SEED + 11703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "voev_homotopy",
            bench_voev_homotopy(seed),
        )
    )


def bench_totaro_cycle_family(
    seed: int = _SEED + 11704,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "totaro_cycle",
            bench_totaro_cycle(seed),
        )
    )


def bench_a1_degrees_family(
    seed: int = _SEED + 11705,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "a1_degrees",
            bench_a1_degrees(seed),
        )
    )
