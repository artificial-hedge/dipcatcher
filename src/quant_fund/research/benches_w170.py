"""Wave-126 adapters: exec-summary causal-DL-2 canon — xlearner,
rlearner, slearner_tlearner, causal_rep_bal, cate_distill, net_drlearner —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cate_distill import bench_cate_distill
from quant_fund.models.causal_rep_bal import bench_causal_rep_bal
from quant_fund.models.net_drlearner import bench_net_drlearner
from quant_fund.models.rlearner import bench_rlearner
from quant_fund.models.slearner_tlearner import bench_slearner_tlearner
from quant_fund.models.xlearner import bench_xlearner

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


def bench_xlearner_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("xlearner", bench_xlearner(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"xlearner bench failed: {exc}") from exc


def bench_rlearner_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("rlearner", bench_rlearner(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rlearner bench failed: {exc}") from exc


def bench_slearner_tlearner_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("slearner_tlearner", bench_slearner_tlearner(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"slearner_tlearner bench failed: {exc}") from exc


def bench_causal_rep_bal_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("causal_rep_bal", bench_causal_rep_bal(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"causal_rep_bal bench failed: {exc}") from exc


def bench_cate_distill_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cate_distill", bench_cate_distill(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cate_distill bench failed: {exc}") from exc


def bench_net_drlearner_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("net_drlearner", bench_net_drlearner(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"net_drlearner bench failed: {exc}") from exc
