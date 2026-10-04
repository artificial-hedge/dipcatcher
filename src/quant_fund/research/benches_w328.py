"""Wave-328 homotopy-type-theory canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.funext_toy import bench_funext_toy
from quant_fund.models.hit_quotient import bench_hit_quotient
from quant_fund.models.hlevel_check import bench_hlevel_check
from quant_fund.models.kan_hcomp import bench_kan_hcomp
from quant_fund.models.path_types import bench_path_types
from quant_fund.models.univalence_toy import bench_univalence_toy

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


def bench_path_types_family(seed: int = _SEED + 1875) -> dict[str, float]:
    return _floats(_finite_blob("path_types", bench_path_types(seed)))


def bench_hlevel_check_family(seed: int = _SEED + 1876) -> dict[str, float]:
    return _floats(_finite_blob("hlevel_check", bench_hlevel_check(seed)))


def bench_univalence_toy_family(seed: int = _SEED + 1877) -> dict[str, float]:
    return _floats(_finite_blob("univalence_toy", bench_univalence_toy(seed)))


def bench_kan_hcomp_family(seed: int = _SEED + 1878) -> dict[str, float]:
    return _floats(_finite_blob("kan_hcomp", bench_kan_hcomp(seed)))


def bench_funext_toy_family(seed: int = _SEED + 1879) -> dict[str, float]:
    return _floats(_finite_blob("funext_toy", bench_funext_toy(seed)))


def bench_hit_quotient_family(seed: int = _SEED + 1880) -> dict[str, float]:
    return _floats(_finite_blob("hit_quotient", bench_hit_quotient(seed)))
