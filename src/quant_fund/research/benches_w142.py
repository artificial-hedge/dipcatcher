"""Wave-126 adapters: exec-summary linear-state-space + adaptive-compute canon — s4_ssm,
rwkv_wkv, hyena_conv, retnet_decay, delta_net, mixture_of_depths —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.delta_net import bench_delta_net
from quant_fund.models.hyena_conv import bench_hyena_conv
from quant_fund.models.mixture_of_depths import bench_mixture_of_depths
from quant_fund.models.retnet_decay import bench_retnet_decay
from quant_fund.models.rwkv_wkv import bench_rwkv_wkv
from quant_fund.models.s4_ssm import bench_s4_ssm

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


def bench_s4_ssm_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("s4_ssm", bench_s4_ssm(seed=_SEED + 840)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"s4_ssm bench failed: {exc}") from exc


def bench_rwkv_wkv_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("rwkv_wkv", bench_rwkv_wkv(seed=_SEED + 841)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rwkv_wkv bench failed: {exc}") from exc


def bench_hyena_conv_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("hyena_conv", bench_hyena_conv(seed=_SEED + 842)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"hyena_conv bench failed: {exc}") from exc


def bench_retnet_decay_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("retnet_decay", bench_retnet_decay(seed=_SEED + 843)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"retnet_decay bench failed: {exc}") from exc


def bench_delta_net_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("delta_net", bench_delta_net(seed=_SEED + 844)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"delta_net bench failed: {exc}") from exc


def bench_mixture_of_depths_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mixture_of_depths", bench_mixture_of_depths(seed=_SEED + 845)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mixture_of_depths bench failed: {exc}") from exc
