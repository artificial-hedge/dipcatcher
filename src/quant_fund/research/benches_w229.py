"""Wave-229 adapters: probabilistic-membership/similarity canon —
bloom_filter, cuckoo_filter, xor_filter, quotient_filter, minhash_lsh,
simhash — benched on SYNTHETIC universes. Adapters flatten to a finite
float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bloom_filter import bench_bloom_filter
from quant_fund.models.cuckoo_filter import bench_cuckoo_filter
from quant_fund.models.minhash_lsh import bench_minhash_lsh
from quant_fund.models.quotient_filter import bench_quotient_filter
from quant_fund.models.simhash import bench_simhash
from quant_fund.models.xor_filter import bench_xor_filter

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


def bench_bloom_filter_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("bloom_filter", bench_bloom_filter(seed=_SEED + 1060)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"bloom_filter bench failed: {exc}") from exc


def bench_cuckoo_filter_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cuckoo_filter", bench_cuckoo_filter(seed=_SEED + 1061)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cuckoo_filter bench failed: {exc}") from exc


def bench_minhash_lsh_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("minhash_lsh", bench_minhash_lsh(seed=_SEED + 1062)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"minhash_lsh bench failed: {exc}") from exc


def bench_quotient_filter_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("quotient_filter", bench_quotient_filter(seed=_SEED + 1063)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"quotient_filter bench failed: {exc}") from exc


def bench_simhash_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("simhash", bench_simhash(seed=_SEED + 1064)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"simhash bench failed: {exc}") from exc


def bench_xor_filter_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("xor_filter", bench_xor_filter(seed=_SEED + 1065)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"xor_filter bench failed: {exc}") from exc
