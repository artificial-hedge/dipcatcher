"""Wave-305 text-index-2/stringology canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.booth_rotation import bench_booth_rotation
from quant_fund.models.lyndon_factor import bench_lyndon_factor
from quant_fund.models.palindromic_tree import bench_palindromic_tree
from quant_fund.models.suffix_array_lcp import bench_suffix_array_lcp
from quant_fund.models.suffix_tree_lex import bench_suffix_tree_lex
from quant_fund.models.z_function import bench_z_function

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


def bench_suffix_array_lcp_family(seed: int = _SEED + 1736) -> dict[str, float]:
    return _floats(_finite_blob("suffix_array_lcp", bench_suffix_array_lcp(seed)))


def bench_z_function_family(seed: int = _SEED + 1737) -> dict[str, float]:
    return _floats(_finite_blob("z_function", bench_z_function(seed)))


def bench_suffix_tree_lex_family(seed: int = _SEED + 1738) -> dict[str, float]:
    return _floats(_finite_blob("suffix_tree_lex", bench_suffix_tree_lex(seed)))


def bench_booth_rotation_family(seed: int = _SEED + 1739) -> dict[str, float]:
    return _floats(_finite_blob("booth_rotation", bench_booth_rotation(seed)))


def bench_lyndon_factor_family(seed: int = _SEED + 1740) -> dict[str, float]:
    return _floats(_finite_blob("lyndon_factor", bench_lyndon_factor(seed)))


def bench_palindromic_tree_family(seed: int = _SEED + 1741) -> dict[str, float]:
    return _floats(_finite_blob("palindromic_tree", bench_palindromic_tree(seed)))
