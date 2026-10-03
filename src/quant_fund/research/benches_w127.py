"""Wave-126 adapters: exec-summary DL SOTA-2 — contrastive_repr,
hypernetwork_alloc, neural_thompson, bnn_ensemble, option_vae, diff_policy —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bnn_ensemble import bench_bnn_ensemble
from quant_fund.models.contrastive_repr import bench_contrastive_repr
from quant_fund.models.diff_policy import bench_diff_policy
from quant_fund.models.hypernetwork_alloc import bench_hypernetwork_alloc
from quant_fund.models.neural_thompson import bench_neural_thompson
from quant_fund.models.option_vae import bench_option_vae

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


def bench_contrastive_repr_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("contrastive_repr", bench_contrastive_repr(seed=_SEED + 750)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"contrastive_repr bench failed: {exc}") from exc


def bench_hypernetwork_alloc_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("hypernetwork_alloc", bench_hypernetwork_alloc(seed=_SEED + 751))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"hypernetwork_alloc bench failed: {exc}") from exc


def bench_neural_thompson_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("neural_thompson", bench_neural_thompson(seed=_SEED + 752)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"neural_thompson bench failed: {exc}") from exc


def bench_bnn_ensemble_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("bnn_ensemble", bench_bnn_ensemble(seed=_SEED + 753)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"bnn_ensemble bench failed: {exc}") from exc


def bench_option_vae_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("option_vae", bench_option_vae(seed=_SEED + 754)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"option_vae bench failed: {exc}") from exc


def bench_diff_policy_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("diff_policy", bench_diff_policy(seed=_SEED + 755)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"diff_policy bench failed: {exc}") from exc
