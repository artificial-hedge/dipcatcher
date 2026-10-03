"""Wave-126 adapters: exec-summary conformal-2 canon — cqr_pred,
survival_cp, aps_cp, ltt_cp, full_cp, risk_cp —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.aps_cp import bench_aps_cp
from quant_fund.models.cqr_pred import bench_cqr_pred
from quant_fund.models.full_cp import bench_full_cp
from quant_fund.models.ltt_cp import bench_ltt_cp
from quant_fund.models.risk_cp import bench_risk_cp
from quant_fund.models.survival_cp import bench_survival_cp

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


def bench_cqr_pred_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cqr_pred", bench_cqr_pred(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cqr_pred bench failed: {exc}") from exc


def bench_survival_cp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("survival_cp", bench_survival_cp(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"survival_cp bench failed: {exc}") from exc


def bench_aps_cp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("aps_cp", bench_aps_cp(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"aps_cp bench failed: {exc}") from exc


def bench_ltt_cp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ltt_cp", bench_ltt_cp(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ltt_cp bench failed: {exc}") from exc


def bench_full_cp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("full_cp", bench_full_cp(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"full_cp bench failed: {exc}") from exc


def bench_risk_cp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("risk_cp", bench_risk_cp(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"risk_cp bench failed: {exc}") from exc
