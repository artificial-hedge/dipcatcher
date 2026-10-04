"""Wave-522 additive-combinatorics bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.freiman_thm import bench_freiman_thm
from quant_fund.models.gowers_norm import bench_gowers_norm
from quant_fund.models.green_tao import bench_green_tao
from quant_fund.models.plunnecke import bench_plunnecke
from quant_fund.models.roth_thm import bench_roth_thm
from quant_fund.models.szemeredi import bench_szemeredi

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


def bench_freiman_thm_family(seed: int = _SEED + 3038) -> dict[str, float]:
    return _floats(_finite_blob("freiman_thm", bench_freiman_thm(seed)))


def bench_szemeredi_family(seed: int = _SEED + 3039) -> dict[str, float]:
    return _floats(_finite_blob("szemeredi", bench_szemeredi(seed)))


def bench_green_tao_family(seed: int = _SEED + 3040) -> dict[str, float]:
    return _floats(_finite_blob("green_tao", bench_green_tao(seed)))


def bench_roth_thm_family(seed: int = _SEED + 3041) -> dict[str, float]:
    return _floats(_finite_blob("roth_thm", bench_roth_thm(seed)))


def bench_gowers_norm_family(seed: int = _SEED + 3042) -> dict[str, float]:
    return _floats(_finite_blob("gowers_norm", bench_gowers_norm(seed)))


def bench_plunnecke_family(seed: int = _SEED + 3043) -> dict[str, float]:
    return _floats(_finite_blob("plunnecke", bench_plunnecke(seed)))
