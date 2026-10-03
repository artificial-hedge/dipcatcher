"""Wave-274 bioinformatics-2 benches: sequence analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.gc_skew import bench_gc_skew
from quant_fund.models.hmm_profile import bench_hmm_profile
from quant_fund.models.kmer_count import bench_kmer_count
from quant_fund.models.orf_find import bench_orf_find
from quant_fund.models.seq_logo import bench_seq_logo
from quant_fund.models.star_msa import bench_star_msa

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


def bench_hmm_profile_family(seed: int = _SEED + 1510) -> dict[str, float]:
    return _floats(_finite_blob("hmm_profile", bench_hmm_profile(seed)))


def bench_star_msa_family(seed: int = _SEED + 1511) -> dict[str, float]:
    return _floats(_finite_blob("star_msa", bench_star_msa(seed)))


def bench_gc_skew_family(seed: int = _SEED + 1512) -> dict[str, float]:
    return _floats(_finite_blob("gc_skew", bench_gc_skew(seed)))


def bench_orf_find_family(seed: int = _SEED + 1513) -> dict[str, float]:
    return _floats(_finite_blob("orf_find", bench_orf_find(seed)))


def bench_kmer_count_family(seed: int = _SEED + 1514) -> dict[str, float]:
    return _floats(_finite_blob("kmer_count", bench_kmer_count(seed)))


def bench_seq_logo_family(seed: int = _SEED + 1515) -> dict[str, float]:
    return _floats(_finite_blob("seq_logo", bench_seq_logo(seed)))
