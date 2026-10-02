"""Wave-126 adapters: exec-summary Bayesian-DL canon — swag_diag,
mc_dropout, bbb_vi, snapshot_ens, concrete_dropout, vcl_online —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bbb_vi import bench_bbb_vi
from quant_fund.models.concrete_dropout import bench_concrete_dropout
from quant_fund.models.mc_dropout import bench_mc_dropout
from quant_fund.models.snapshot_ens import bench_snapshot_ens
from quant_fund.models.swag_diag import bench_swag_diag
from quant_fund.models.vcl_online import bench_vcl_online

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


def bench_swag_diag_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("swag_diag", bench_swag_diag(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"swag_diag bench failed: {exc}") from exc


def bench_mc_dropout_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mc_dropout", bench_mc_dropout(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mc_dropout bench failed: {exc}") from exc


def bench_bbb_vi_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("bbb_vi", bench_bbb_vi(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"bbb_vi bench failed: {exc}") from exc


def bench_snapshot_ens_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("snapshot_ens", bench_snapshot_ens(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"snapshot_ens bench failed: {exc}") from exc


def bench_concrete_dropout_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("concrete_dropout", bench_concrete_dropout(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"concrete_dropout bench failed: {exc}") from exc


def bench_vcl_online_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("vcl_online", bench_vcl_online(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"vcl_online bench failed: {exc}") from exc
