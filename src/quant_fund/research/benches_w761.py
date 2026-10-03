"""Wave-761 renewal-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.alternating_renewal import (
    bench_alternating_renewal,
)
from quant_fund.models.blackwell_renewal import (
    bench_blackwell_renewal,
)
from quant_fund.models.delayed_renewal import bench_delayed_renewal
from quant_fund.models.excess_renewal import bench_excess_renewal
from quant_fund.models.key_renewal import bench_key_renewal
from quant_fund.models.renewal_reward2 import (
    bench_renewal_reward2,
)

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


def bench_blackwell_renewal_family(
    seed: int = _SEED + 15000,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "blackwell_renewal",
            bench_blackwell_renewal(seed),
        )
    )


def bench_key_renewal_family(
    seed: int = _SEED + 15001,
) -> dict[str, float]:
    return _floats(_finite_blob("key_renewal", bench_key_renewal(seed)))


def bench_excess_renewal_family(
    seed: int = _SEED + 15002,
) -> dict[str, float]:
    return _floats(_finite_blob("excess_renewal", bench_excess_renewal(seed)))


def bench_alternating_renewal_family(
    seed: int = _SEED + 15003,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "alternating_renewal",
            bench_alternating_renewal(seed),
        )
    )


def bench_renewal_reward2_family(
    seed: int = _SEED + 15004,
) -> dict[str, float]:
    return _floats(_finite_blob("renewal_reward2", bench_renewal_reward2(seed)))


def bench_delayed_renewal_family(
    seed: int = _SEED + 15005,
) -> dict[str, float]:
    return _floats(_finite_blob("delayed_renewal", bench_delayed_renewal(seed)))
