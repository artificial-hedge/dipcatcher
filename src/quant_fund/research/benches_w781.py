"""Wave-781 regeneration/Khinchin bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.karlin_mcg import bench_karlin_mcg
from quant_fund.models.keilson_stieltjes import (
    bench_keilson_stieltjes,
)
from quant_fund.models.korolyuk import bench_korolyuk
from quant_fund.models.palm_khinchin import (
    bench_palm_khinchin,
)
from quant_fund.models.regen_proc import bench_regen_proc
from quant_fund.models.wold_proc import bench_wold_proc

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


def bench_karlin_mcg_family(
    seed: int = _SEED + 17000,
) -> dict[str, float]:
    return _floats(_finite_blob("karlin_mcg", bench_karlin_mcg(seed)))


def bench_keilson_stieltjes_family(
    seed: int = _SEED + 17001,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "keilson_stieltjes",
            bench_keilson_stieltjes(seed),
        )
    )


def bench_palm_khinchin_family(
    seed: int = _SEED + 17002,
) -> dict[str, float]:
    return _floats(_finite_blob("palm_khinchin", bench_palm_khinchin(seed)))


def bench_regen_proc_family(
    seed: int = _SEED + 17003,
) -> dict[str, float]:
    return _floats(_finite_blob("regen_proc", bench_regen_proc(seed)))


def bench_wold_proc_family(
    seed: int = _SEED + 17004,
) -> dict[str, float]:
    return _floats(_finite_blob("wold_proc", bench_wold_proc(seed)))


def bench_korolyuk_family(
    seed: int = _SEED + 17005,
) -> dict[str, float]:
    return _floats(_finite_blob("korolyuk", bench_korolyuk(seed)))
