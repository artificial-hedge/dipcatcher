"""Wave-126 adapters: exec-summary causal-discovery classical canon — var_lingam,
ges_search, direct_lingam, fci_alg, ica_lingam, mmmb_select —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.direct_lingam import bench_direct_lingam
from quant_fund.models.fci_alg import bench_fci_alg
from quant_fund.models.ges_search import bench_ges_search
from quant_fund.models.ica_lingam import bench_ica_lingam
from quant_fund.models.mmmb_select import bench_mmmb_select
from quant_fund.models.var_lingam import bench_var_lingam

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


def bench_var_lingam_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("var_lingam", bench_var_lingam(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"var_lingam bench failed: {exc}") from exc


def bench_ges_search_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ges_search", bench_ges_search(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ges_search bench failed: {exc}") from exc


def bench_direct_lingam_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("direct_lingam", bench_direct_lingam(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"direct_lingam bench failed: {exc}") from exc


def bench_fci_alg_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("fci_alg", bench_fci_alg(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"fci_alg bench failed: {exc}") from exc


def bench_ica_lingam_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ica_lingam", bench_ica_lingam(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ica_lingam bench failed: {exc}") from exc


def bench_mmmb_select_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mmmb_select", bench_mmmb_select(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mmmb_select bench failed: {exc}") from exc
