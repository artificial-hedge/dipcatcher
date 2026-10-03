"""Wave-126 adapters: exec-summary causal-DL canon — tarnet_ite,
dragonnet_dr, deep_iv, cevae_latent, causal_rep, policy_value —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.causal_rep import bench_causal_rep
from quant_fund.models.cevae_latent import bench_cevae_latent
from quant_fund.models.deep_iv import bench_deep_iv
from quant_fund.models.dragonnet_dr import bench_dragonnet_dr
from quant_fund.models.policy_value import bench_policy_value
from quant_fund.models.tarnet_ite import bench_tarnet_ite

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


def bench_tarnet_ite_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("tarnet_ite", bench_tarnet_ite(seed=_SEED + 918)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tarnet_ite bench failed: {exc}") from exc


def bench_dragonnet_dr_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("dragonnet_dr", bench_dragonnet_dr(seed=_SEED + 919)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dragonnet_dr bench failed: {exc}") from exc


def bench_deep_iv_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("deep_iv", bench_deep_iv(seed=_SEED + 920)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"deep_iv bench failed: {exc}") from exc


def bench_cevae_latent_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cevae_latent", bench_cevae_latent(seed=_SEED + 921)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cevae_latent bench failed: {exc}") from exc


def bench_causal_rep_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("causal_rep", bench_causal_rep(seed=_SEED + 922)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"causal_rep bench failed: {exc}") from exc


def bench_policy_value_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("policy_value", bench_policy_value(seed=_SEED + 923)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"policy_value bench failed: {exc}") from exc
