"""Wave-581 spectral-sequences bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bockstein_ss import bench_bockstein_ss
from quant_fund.models.bousfield_ss import bench_bousfield_ss
from quant_fund.models.cartan_ss import bench_cartan_ss
from quant_fund.models.eilenberg_moore import (
    bench_eilenberg_moore,
)
from quant_fund.models.lyndon_ss import bench_lyndon_ss
from quant_fund.models.serre_ss4 import bench_serre_ss4

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


def bench_serre_ss4_family(seed: int = _SEED + 3392) -> dict[str, float]:
    return _floats(_finite_blob("serre_ss4", bench_serre_ss4(seed)))


def bench_bockstein_ss_family(seed: int = _SEED + 3393) -> dict[str, float]:
    return _floats(_finite_blob("bockstein_ss", bench_bockstein_ss(seed)))


def bench_eilenberg_moore_family(
    seed: int = _SEED + 3394,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "eilenberg_moore",
            bench_eilenberg_moore(seed),
        )
    )


def bench_bousfield_ss_family(seed: int = _SEED + 3395) -> dict[str, float]:
    return _floats(_finite_blob("bousfield_ss", bench_bousfield_ss(seed)))


def bench_lyndon_ss_family(seed: int = _SEED + 3396) -> dict[str, float]:
    return _floats(_finite_blob("lyndon_ss", bench_lyndon_ss(seed)))


def bench_cartan_ss_family(seed: int = _SEED + 3397) -> dict[str, float]:
    return _floats(_finite_blob("cartan_ss", bench_cartan_ss(seed)))
