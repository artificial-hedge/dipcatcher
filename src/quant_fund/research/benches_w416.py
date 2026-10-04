"""Wave-416 model-theory-6 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.decidable_theory import bench_decidable_theory
from quant_fund.models.definable_set import bench_definable_set
from quant_fund.models.indiscernible_seq import bench_indiscernible_seq
from quant_fund.models.interpol_thm import bench_interpol_thm
from quant_fund.models.omitting_prime import bench_omitting_prime
from quant_fund.models.saturated_model import bench_saturated_model

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


def bench_decidable_theory_family(
    seed: int = _SEED + 2402,
) -> dict[str, float]:
    return _floats(_finite_blob("decidable_theory", bench_decidable_theory(seed)))


def bench_indiscernible_seq_family(
    seed: int = _SEED + 2403,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "indiscernible_seq",
            bench_indiscernible_seq(seed),
        )
    )


def bench_saturated_model_family(
    seed: int = _SEED + 2404,
) -> dict[str, float]:
    return _floats(_finite_blob("saturated_model", bench_saturated_model(seed)))


def bench_omitting_prime_family(
    seed: int = _SEED + 2405,
) -> dict[str, float]:
    return _floats(_finite_blob("omitting_prime", bench_omitting_prime(seed)))


def bench_interpol_thm_family(
    seed: int = _SEED + 2406,
) -> dict[str, float]:
    return _floats(_finite_blob("interpol_thm", bench_interpol_thm(seed)))


def bench_definable_set_family(
    seed: int = _SEED + 2407,
) -> dict[str, float]:
    return _floats(_finite_blob("definable_set", bench_definable_set(seed)))
