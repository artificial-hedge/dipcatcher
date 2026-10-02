"""Wave-126 adapters: exec-summary learning-to-rank canon — ranknet_ltr,
listnet_ltr, listmle_ltr, lambdarank_ltr, approx_ndcg_ltr, neural_sort_ltr —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.approx_ndcg_ltr import bench_approx_ndcg_ltr
from quant_fund.models.lambdarank_ltr import bench_lambdarank_ltr
from quant_fund.models.listmle_ltr import bench_listmle_ltr
from quant_fund.models.listnet_ltr import bench_listnet_ltr
from quant_fund.models.neural_sort_ltr import bench_neural_sort_ltr
from quant_fund.models.ranknet_ltr import bench_ranknet_ltr

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


def bench_ranknet_ltr_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ranknet_ltr", bench_ranknet_ltr(seed=_SEED + 936)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ranknet_ltr bench failed: {exc}") from exc


def bench_listnet_ltr_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("listnet_ltr", bench_listnet_ltr(seed=_SEED + 937)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"listnet_ltr bench failed: {exc}") from exc


def bench_listmle_ltr_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("listmle_ltr", bench_listmle_ltr(seed=_SEED + 938)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"listmle_ltr bench failed: {exc}") from exc


def bench_lambdarank_ltr_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lambdarank_ltr", bench_lambdarank_ltr(seed=_SEED + 939)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lambdarank_ltr bench failed: {exc}") from exc


def bench_approx_ndcg_ltr_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("approx_ndcg_ltr", bench_approx_ndcg_ltr(seed=_SEED + 940)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"approx_ndcg_ltr bench failed: {exc}") from exc


def bench_neural_sort_ltr_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("neural_sort_ltr", bench_neural_sort_ltr(seed=_SEED + 941)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"neural_sort_ltr bench failed: {exc}") from exc
