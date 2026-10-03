"""Wave-282 combinatorics benches: counting, codes, Ramsey, Latin squares."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.gray_code import bench_gray_code
from quant_fund.models.inversion_count import bench_inversion_count
from quant_fund.models.latin_square import bench_latin_square
from quant_fund.models.ramsey_bound import bench_ramsey_bound
from quant_fund.models.stirling_count import bench_stirling_count
from quant_fund.models.subset_sum_dp import bench_subset_sum_dp

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


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


def bench_subset_sum_dp_family(seed: int = _SEED + 1590) -> dict[str, float]:
    return _floats(_finite_blob("subset_sum_dp", bench_subset_sum_dp(seed)))


def bench_stirling_count_family(seed: int = _SEED + 1591) -> dict[str, float]:
    return _floats(_finite_blob("stirling_count", bench_stirling_count(seed)))


def bench_gray_code_family(seed: int = _SEED + 1592) -> dict[str, float]:
    return _floats(_finite_blob("gray_code", bench_gray_code(seed)))


def bench_inversion_count_family(seed: int = _SEED + 1593) -> dict[str, float]:
    return _floats(_finite_blob("inversion_count", bench_inversion_count(seed)))


def bench_ramsey_bound_family(seed: int = _SEED + 1594) -> dict[str, float]:
    return _floats(_finite_blob("ramsey_bound", bench_ramsey_bound(seed)))


def bench_latin_square_family(seed: int = _SEED + 1595) -> dict[str, float]:
    return _floats(_finite_blob("latin_square", bench_latin_square(seed)))
