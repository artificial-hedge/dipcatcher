"""Wave-371 differential-topology canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.degree_mod2 import bench_degree_mod2
from quant_fund.models.handle_decomp import bench_handle_decomp
from quant_fund.models.morse_theory import bench_morse_theory
from quant_fund.models.poincare_hopf import bench_poincare_hopf
from quant_fund.models.regular_value import bench_regular_value
from quant_fund.models.transversality import bench_transversality

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


def bench_morse_theory_family(seed: int = _SEED + 2132) -> dict[str, float]:
    return _floats(_finite_blob("morse_theory", bench_morse_theory(seed)))


def bench_transversality_family(seed: int = _SEED + 2133) -> dict[str, float]:
    return _floats(_finite_blob("transversality", bench_transversality(seed)))


def bench_regular_value_family(seed: int = _SEED + 2134) -> dict[str, float]:
    return _floats(_finite_blob("regular_value", bench_regular_value(seed)))


def bench_degree_mod2_family(seed: int = _SEED + 2135) -> dict[str, float]:
    return _floats(_finite_blob("degree_mod2", bench_degree_mod2(seed)))


def bench_handle_decomp_family(seed: int = _SEED + 2136) -> dict[str, float]:
    return _floats(_finite_blob("handle_decomp", bench_handle_decomp(seed)))


def bench_poincare_hopf_family(seed: int = _SEED + 2137) -> dict[str, float]:
    return _floats(_finite_blob("poincare_hopf", bench_poincare_hopf(seed)))
