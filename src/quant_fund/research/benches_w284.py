"""Wave-284 bioinformatics-3 canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.band_align import bench_band_align
from quant_fund.models.codon_usage import bench_codon_usage
from quant_fund.models.fitch_pars import bench_fitch_pars
from quant_fund.models.jc69_lik import bench_jc69_lik
from quant_fund.models.nj_tree import bench_nj_tree
from quant_fund.models.seed_extend import bench_seed_extend

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


def bench_nj_tree_family(seed: int = _SEED + 1610) -> dict[str, float]:
    return _floats(_finite_blob("nj_tree", bench_nj_tree(seed)))


def bench_fitch_pars_family(seed: int = _SEED + 1611) -> dict[str, float]:
    return _floats(_finite_blob("fitch_pars", bench_fitch_pars(seed)))


def bench_seed_extend_family(seed: int = _SEED + 1612) -> dict[str, float]:
    return _floats(_finite_blob("seed_extend", bench_seed_extend(seed)))


def bench_band_align_family(seed: int = _SEED + 1613) -> dict[str, float]:
    return _floats(_finite_blob("band_align", bench_band_align(seed)))


def bench_jc69_lik_family(seed: int = _SEED + 1614) -> dict[str, float]:
    return _floats(_finite_blob("jc69_lik", bench_jc69_lik(seed)))


def bench_codon_usage_family(seed: int = _SEED + 1615) -> dict[str, float]:
    return _floats(_finite_blob("codon_usage", bench_codon_usage(seed)))
