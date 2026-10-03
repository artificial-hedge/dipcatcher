"""Wave-340 homological-algebra/algebraic-geometry canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chain_complex import bench_chain_complex
from quant_fund.models.hilbert_series import bench_hilbert_series
from quant_fund.models.sheaf_check import bench_sheaf_check
from quant_fund.models.snake_lemma import bench_snake_lemma
from quant_fund.models.tor_ext import bench_tor_ext
from quant_fund.models.variety_morph import bench_variety_morph

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


def bench_chain_complex_family(seed: int = _SEED + 1947) -> dict[str, float]:
    return _floats(_finite_blob("chain_complex", bench_chain_complex(seed)))


def bench_tor_ext_family(seed: int = _SEED + 1948) -> dict[str, float]:
    return _floats(_finite_blob("tor_ext", bench_tor_ext(seed)))


def bench_sheaf_check_family(seed: int = _SEED + 1949) -> dict[str, float]:
    return _floats(_finite_blob("sheaf_check", bench_sheaf_check(seed)))


def bench_hilbert_series_family(seed: int = _SEED + 1950) -> dict[str, float]:
    return _floats(_finite_blob("hilbert_series", bench_hilbert_series(seed)))


def bench_snake_lemma_family(seed: int = _SEED + 1951) -> dict[str, float]:
    return _floats(_finite_blob("snake_lemma", bench_snake_lemma(seed)))


def bench_variety_morph_family(seed: int = _SEED + 1952) -> dict[str, float]:
    return _floats(_finite_blob("variety_morph", bench_variety_morph(seed)))
