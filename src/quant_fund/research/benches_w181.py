"""Wave-126 adapters: exec-summary LM-arch-2 canon — mamba2_ssd,
xlstm_mlstm, rwkv7, titans_memory, gated_deltanet, longhorn_ssm —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.gated_deltanet import bench_gated_deltanet
from quant_fund.models.longhorn_ssm import bench_longhorn_ssm
from quant_fund.models.mamba2_ssd import bench_mamba2_ssd
from quant_fund.models.rwkv7 import bench_rwkv7
from quant_fund.models.titans_memory import bench_titans_memory
from quant_fund.models.xlstm_mlstm import bench_xlstm_mlstm

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


def bench_mamba2_ssd_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mamba2_ssd", bench_mamba2_ssd(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mamba2_ssd bench failed: {exc}") from exc


def bench_xlstm_mlstm_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("xlstm_mlstm", bench_xlstm_mlstm(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"xlstm_mlstm bench failed: {exc}") from exc


def bench_rwkv7_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("rwkv7", bench_rwkv7(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rwkv7 bench failed: {exc}") from exc


def bench_titans_memory_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("titans_memory", bench_titans_memory(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"titans_memory bench failed: {exc}") from exc


def bench_gated_deltanet_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gated_deltanet", bench_gated_deltanet(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gated_deltanet bench failed: {exc}") from exc


def bench_longhorn_ssm_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("longhorn_ssm", bench_longhorn_ssm(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"longhorn_ssm bench failed: {exc}") from exc
