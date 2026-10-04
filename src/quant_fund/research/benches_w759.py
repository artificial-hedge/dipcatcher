"""Wave-759 CLE-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.apu_cle import bench_apu_cle
from quant_fund.models.gwynne_cle import bench_gwynne_cle
from quant_fund.models.hospitsky_cle import bench_hospitsky_cle
from quant_fund.models.nolin_cle import bench_nolin_cle
from quant_fund.models.sun_cle import bench_sun_cle
from quant_fund.models.zhan_cle import bench_zhan_cle

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


def bench_gwynne_cle_family(
    seed: int = _SEED + 14800,
) -> dict[str, float]:
    return _floats(_finite_blob("gwynne_cle", bench_gwynne_cle(seed)))


def bench_hospitsky_cle_family(
    seed: int = _SEED + 14801,
) -> dict[str, float]:
    return _floats(_finite_blob("hospitsky_cle", bench_hospitsky_cle(seed)))


def bench_apu_cle_family(
    seed: int = _SEED + 14802,
) -> dict[str, float]:
    return _floats(_finite_blob("apu_cle", bench_apu_cle(seed)))


def bench_nolin_cle_family(
    seed: int = _SEED + 14803,
) -> dict[str, float]:
    return _floats(_finite_blob("nolin_cle", bench_nolin_cle(seed)))


def bench_sun_cle_family(
    seed: int = _SEED + 14804,
) -> dict[str, float]:
    return _floats(_finite_blob("sun_cle", bench_sun_cle(seed)))


def bench_zhan_cle_family(
    seed: int = _SEED + 14805,
) -> dict[str, float]:
    return _floats(_finite_blob("zhan_cle", bench_zhan_cle(seed)))
