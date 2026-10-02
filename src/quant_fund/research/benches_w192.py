"""Wave-126 adapters: exec-summary info-theory canon — copula_mi,
mmd_two_sample, nwj_mi, hsic_independence, mine_mi, lsd_deptest —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.copula_mi import bench_copula_mi
from quant_fund.models.hsic_independence import bench_hsic_independence
from quant_fund.models.lsd_deptest import bench_lsd_deptest
from quant_fund.models.mine_mi import bench_mine_mi
from quant_fund.models.mmd_two_sample import bench_mmd_two_sample
from quant_fund.models.nwj_mi import bench_nwj_mi

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


def bench_copula_mi_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("copula_mi", bench_copula_mi(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"copula_mi bench failed: {exc}") from exc


def bench_mmd_two_sample_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mmd_two_sample", bench_mmd_two_sample(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mmd_two_sample bench failed: {exc}") from exc


def bench_nwj_mi_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("nwj_mi", bench_nwj_mi(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"nwj_mi bench failed: {exc}") from exc


def bench_hsic_independence_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("hsic_independence", bench_hsic_independence(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"hsic_independence bench failed: {exc}") from exc


def bench_mine_mi_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mine_mi", bench_mine_mi(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mine_mi bench failed: {exc}") from exc


def bench_lsd_deptest_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lsd_deptest", bench_lsd_deptest(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lsd_deptest bench failed: {exc}") from exc
