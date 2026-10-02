"""Wave-126 adapters: exec-summary graph-temporal canon — dcrnn_lite,
stgcn_lite, gwnet_lite, astgcn, mtgnn_lite, agcrn —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.agcrn import bench_agcrn
from quant_fund.models.astgcn import bench_astgcn
from quant_fund.models.dcrnn_lite import bench_dcrnn_lite
from quant_fund.models.gwnet_lite import bench_gwnet_lite
from quant_fund.models.mtgnn_lite import bench_mtgnn_lite
from quant_fund.models.stgcn_lite import bench_stgcn_lite

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


def bench_dcrnn_lite_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("dcrnn_lite", bench_dcrnn_lite(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dcrnn_lite bench failed: {exc}") from exc


def bench_stgcn_lite_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("stgcn_lite", bench_stgcn_lite(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"stgcn_lite bench failed: {exc}") from exc


def bench_gwnet_lite_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gwnet_lite", bench_gwnet_lite(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gwnet_lite bench failed: {exc}") from exc


def bench_astgcn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("astgcn", bench_astgcn(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"astgcn bench failed: {exc}") from exc


def bench_mtgnn_lite_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mtgnn_lite", bench_mtgnn_lite(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mtgnn_lite bench failed: {exc}") from exc


def bench_agcrn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("agcrn", bench_agcrn(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"agcrn bench failed: {exc}") from exc
