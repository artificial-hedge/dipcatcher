"""Wave-753 O(N)-model bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.aizenman_irf import bench_aizenman_irf
from quant_fund.models.cardy_on import bench_cardy_on
from quant_fund.models.fernandez_frohlich import (
    bench_fernandez_frohlich,
)
from quant_fund.models.fradkin_sokal import bench_fradkin_sokal
from quant_fund.models.nienhuis_on import bench_nienhuis_on
from quant_fund.models.pelissetto_vicari import (
    bench_pelissetto_vicari,
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


def bench_fernandez_frohlich_family(
    seed: int = _SEED + 14200,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fernandez_frohlich",
            bench_fernandez_frohlich(seed),
        )
    )


def bench_aizenman_irf_family(
    seed: int = _SEED + 14201,
) -> dict[str, float]:
    return _floats(_finite_blob("aizenman_irf", bench_aizenman_irf(seed)))


def bench_fradkin_sokal_family(
    seed: int = _SEED + 14202,
) -> dict[str, float]:
    return _floats(_finite_blob("fradkin_sokal", bench_fradkin_sokal(seed)))


def bench_nienhuis_on_family(
    seed: int = _SEED + 14203,
) -> dict[str, float]:
    return _floats(_finite_blob("nienhuis_on", bench_nienhuis_on(seed)))


def bench_cardy_on_family(
    seed: int = _SEED + 14204,
) -> dict[str, float]:
    return _floats(_finite_blob("cardy_on", bench_cardy_on(seed)))


def bench_pelissetto_vicari_family(
    seed: int = _SEED + 14205,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "pelissetto_vicari",
            bench_pelissetto_vicari(seed),
        )
    )
