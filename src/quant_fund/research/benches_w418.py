"""Wave-418 probability-5 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.borel_cantelli import bench_borel_cantelli
from quant_fund.models.clt_classic import bench_clt_classic
from quant_fund.models.dominated_conv import bench_dominated_conv
from quant_fund.models.strong_lln import bench_strong_lln
from quant_fund.models.uniform_lln import bench_uniform_lln
from quant_fund.models.weak_law import bench_weak_law

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


def bench_weak_law_family(seed: int = _SEED + 2414) -> dict[str, float]:
    return _floats(_finite_blob("weak_law", bench_weak_law(seed)))


def bench_strong_lln_family(
    seed: int = _SEED + 2415,
) -> dict[str, float]:
    return _floats(_finite_blob("strong_lln", bench_strong_lln(seed)))


def bench_clt_classic_family(
    seed: int = _SEED + 2416,
) -> dict[str, float]:
    return _floats(_finite_blob("clt_classic", bench_clt_classic(seed)))


def bench_borel_cantelli_family(
    seed: int = _SEED + 2417,
) -> dict[str, float]:
    return _floats(_finite_blob("borel_cantelli", bench_borel_cantelli(seed)))


def bench_dominated_conv_family(
    seed: int = _SEED + 2418,
) -> dict[str, float]:
    return _floats(_finite_blob("dominated_conv", bench_dominated_conv(seed)))


def bench_uniform_lln_family(
    seed: int = _SEED + 2419,
) -> dict[str, float]:
    return _floats(_finite_blob("uniform_lln", bench_uniform_lln(seed)))
