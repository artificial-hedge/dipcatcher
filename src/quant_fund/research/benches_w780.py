"""Wave-780 queueing-network bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bcmp_net import bench_bcmp_net
from quant_fund.models.convoy_net import bench_convoy_net
from quant_fund.models.insensitive_thm import (
    bench_insensitive_thm,
)
from quant_fund.models.kaufman_roberts import (
    bench_kaufman_roberts,
)
from quant_fund.models.mean_value import bench_mean_value
from quant_fund.models.orku_loss import bench_orku_loss

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


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


def bench_bcmp_net_family(
    seed: int = _SEED + 16900,
) -> dict[str, float]:
    return _floats(_finite_blob("bcmp_net", bench_bcmp_net(seed)))


def bench_mean_value_family(
    seed: int = _SEED + 16901,
) -> dict[str, float]:
    return _floats(_finite_blob("mean_value", bench_mean_value(seed)))


def bench_convoy_net_family(
    seed: int = _SEED + 16902,
) -> dict[str, float]:
    return _floats(_finite_blob("convoy_net", bench_convoy_net(seed)))


def bench_insensitive_thm_family(
    seed: int = _SEED + 16903,
) -> dict[str, float]:
    return _floats(_finite_blob("insensitive_thm", bench_insensitive_thm(seed)))


def bench_kaufman_roberts_family(
    seed: int = _SEED + 16904,
) -> dict[str, float]:
    return _floats(_finite_blob("kaufman_roberts", bench_kaufman_roberts(seed)))


def bench_orku_loss_family(
    seed: int = _SEED + 16905,
) -> dict[str, float]:
    return _floats(_finite_blob("orku_loss", bench_orku_loss(seed)))
