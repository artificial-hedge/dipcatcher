"""Wave-126 adapters: exec-summary active-learning canon — badge_embed,
entropy_query, coreset_kcenter, margin_sampling, qbc_committee, egl_change —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.badge_embed import bench_badge_embed
from quant_fund.models.coreset_kcenter import bench_coreset_kcenter
from quant_fund.models.egl_change import bench_egl_change
from quant_fund.models.entropy_query import bench_entropy_query
from quant_fund.models.margin_sampling import bench_margin_sampling
from quant_fund.models.qbc_committee import bench_qbc_committee

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


def bench_badge_embed_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("badge_embed", bench_badge_embed(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"badge_embed bench failed: {exc}") from exc


def bench_entropy_query_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("entropy_query", bench_entropy_query(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"entropy_query bench failed: {exc}") from exc


def bench_coreset_kcenter_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("coreset_kcenter", bench_coreset_kcenter(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"coreset_kcenter bench failed: {exc}") from exc


def bench_margin_sampling_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("margin_sampling", bench_margin_sampling(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"margin_sampling bench failed: {exc}") from exc


def bench_qbc_committee_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("qbc_committee", bench_qbc_committee(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"qbc_committee bench failed: {exc}") from exc


def bench_egl_change_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("egl_change", bench_egl_change(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"egl_change bench failed: {exc}") from exc
