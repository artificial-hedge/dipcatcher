"""Wave-527 hyperbolic-dynamics bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.anosov import bench_anosov
from quant_fund.models.bowen_spec import bench_bowen_spec
from quant_fund.models.horseshoe import bench_horseshoe
from quant_fund.models.markov_partition import bench_markov_partition
from quant_fund.models.srb_measure import bench_srb_measure
from quant_fund.models.stable_mfld import bench_stable_mfld

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


def bench_anosov_family(seed: int = _SEED + 3068) -> dict[str, float]:
    return _floats(_finite_blob("anosov", bench_anosov(seed)))


def bench_srb_measure_family(seed: int = _SEED + 3069) -> dict[str, float]:
    return _floats(_finite_blob("srb_measure", bench_srb_measure(seed)))


def bench_horseshoe_family(seed: int = _SEED + 3070) -> dict[str, float]:
    return _floats(_finite_blob("horseshoe", bench_horseshoe(seed)))


def bench_stable_mfld_family(seed: int = _SEED + 3071) -> dict[str, float]:
    return _floats(_finite_blob("stable_mfld", bench_stable_mfld(seed)))


def bench_bowen_spec_family(seed: int = _SEED + 3072) -> dict[str, float]:
    return _floats(_finite_blob("bowen_spec", bench_bowen_spec(seed)))


def bench_markov_partition_family(
    seed: int = _SEED + 3073,
) -> dict[str, float]:
    return _floats(_finite_blob("markov_partition", bench_markov_partition(seed)))
