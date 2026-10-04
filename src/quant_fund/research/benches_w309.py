"""Wave-309 post-quantum-3 canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bike_lite import bench_bike_lite
from quant_fund.models.hqc_lite import bench_hqc_lite
from quant_fund.models.mceliece_lite import bench_mceliece_lite
from quant_fund.models.rainbow_sig import bench_rainbow_sig
from quant_fund.models.sidh_lite import bench_sidh_lite
from quant_fund.models.uov_sig import bench_uov_sig

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


def bench_mceliece_lite_family(seed: int = _SEED + 1760) -> dict[str, float]:
    return _floats(_finite_blob("mceliece_lite", bench_mceliece_lite(seed)))


def bench_bike_lite_family(seed: int = _SEED + 1761) -> dict[str, float]:
    return _floats(_finite_blob("bike_lite", bench_bike_lite(seed)))


def bench_hqc_lite_family(seed: int = _SEED + 1762) -> dict[str, float]:
    return _floats(_finite_blob("hqc_lite", bench_hqc_lite(seed)))


def bench_uov_sig_family(seed: int = _SEED + 1763) -> dict[str, float]:
    return _floats(_finite_blob("uov_sig", bench_uov_sig(seed)))


def bench_rainbow_sig_family(seed: int = _SEED + 1764) -> dict[str, float]:
    return _floats(_finite_blob("rainbow_sig", bench_rainbow_sig(seed)))


def bench_sidh_lite_family(seed: int = _SEED + 1765) -> dict[str, float]:
    return _floats(_finite_blob("sidh_lite", bench_sidh_lite(seed)))
