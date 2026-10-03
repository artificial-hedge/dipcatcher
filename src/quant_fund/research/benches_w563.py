"""Wave-563 harmonic-maps bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bubbling_hm import bench_bubbling_hm
from quant_fund.models.eells_sampson import bench_eells_sampson
from quant_fund.models.harmonic_map import bench_harmonic_map
from quant_fund.models.heat_flow_hm import bench_heat_flow_hm
from quant_fund.models.sacks_uhlenbeck import bench_sacks_uhlenbeck
from quant_fund.models.schoen_uhlenbeck import bench_schoen_uhlenbeck

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


def bench_harmonic_map_family(seed: int = _SEED + 3284) -> dict[str, float]:
    return _floats(_finite_blob("harmonic_map", bench_harmonic_map(seed)))


def bench_eells_sampson_family(seed: int = _SEED + 3285) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "eells_sampson",
            bench_eells_sampson(seed),
        )
    )


def bench_schoen_uhlenbeck_family(
    seed: int = _SEED + 3286,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "schoen_uhlenbeck",
            bench_schoen_uhlenbeck(seed),
        )
    )


def bench_bubbling_hm_family(seed: int = _SEED + 3287) -> dict[str, float]:
    return _floats(_finite_blob("bubbling_hm", bench_bubbling_hm(seed)))


def bench_heat_flow_hm_family(seed: int = _SEED + 3288) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "heat_flow_hm",
            bench_heat_flow_hm(seed),
        )
    )


def bench_sacks_uhlenbeck_family(
    seed: int = _SEED + 3289,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "sacks_uhlenbeck",
            bench_sacks_uhlenbeck(seed),
        )
    )
