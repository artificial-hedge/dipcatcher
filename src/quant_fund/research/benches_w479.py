"""Wave-479 p-adic-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ahb_ring import bench_ahb_ring
from quant_fund.models.drinfeld_sym import bench_drinfeld_sym
from quant_fund.models.fargues_diam import bench_fargues_diam
from quant_fund.models.prism_2 import bench_prism_2
from quant_fund.models.scholze_diamond import bench_scholze_diamond
from quant_fund.models.tilting_equiv import bench_tilting_equiv

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


def bench_fargues_diam_family(seed: int = _SEED + 2780) -> dict[str, float]:
    return _floats(_finite_blob("fargues_diam", bench_fargues_diam(seed)))


def bench_tilting_equiv_family(seed: int = _SEED + 2781) -> dict[str, float]:
    return _floats(_finite_blob("tilting_equiv", bench_tilting_equiv(seed)))


def bench_scholze_diamond_family(seed: int = _SEED + 2782) -> dict[str, float]:
    return _floats(_finite_blob("scholze_diamond", bench_scholze_diamond(seed)))


def bench_ahb_ring_family(seed: int = _SEED + 2783) -> dict[str, float]:
    return _floats(_finite_blob("ahb_ring", bench_ahb_ring(seed)))


def bench_prism_2_family(seed: int = _SEED + 2784) -> dict[str, float]:
    return _floats(_finite_blob("prism_2", bench_prism_2(seed)))


def bench_drinfeld_sym_family(seed: int = _SEED + 2785) -> dict[str, float]:
    return _floats(_finite_blob("drinfeld_sym", bench_drinfeld_sym(seed)))
