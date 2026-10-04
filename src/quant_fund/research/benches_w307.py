"""Wave-307 regex-2 canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bitap_fuzzy import bench_bitap_fuzzy
from quant_fund.models.glushkov_nfa import bench_glushkov_nfa
from quant_fund.models.lazy_dfa import bench_lazy_dfa
from quant_fund.models.literal_prefilter import bench_literal_prefilter
from quant_fund.models.pike_vm import bench_pike_vm
from quant_fund.models.regex_simplify import bench_regex_simplify

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


def bench_pike_vm_family(seed: int = _SEED + 1748) -> dict[str, float]:
    return _floats(_finite_blob("pike_vm", bench_pike_vm(seed)))


def bench_lazy_dfa_family(seed: int = _SEED + 1749) -> dict[str, float]:
    return _floats(_finite_blob("lazy_dfa", bench_lazy_dfa(seed)))


def bench_bitap_fuzzy_family(seed: int = _SEED + 1750) -> dict[str, float]:
    return _floats(_finite_blob("bitap_fuzzy", bench_bitap_fuzzy(seed)))


def bench_literal_prefilter_family(seed: int = _SEED + 1751) -> dict[str, float]:
    return _floats(_finite_blob("literal_prefilter", bench_literal_prefilter(seed)))


def bench_glushkov_nfa_family(seed: int = _SEED + 1752) -> dict[str, float]:
    return _floats(_finite_blob("glushkov_nfa", bench_glushkov_nfa(seed)))


def bench_regex_simplify_family(seed: int = _SEED + 1753) -> dict[str, float]:
    return _floats(_finite_blob("regex_simplify", bench_regex_simplify(seed)))
