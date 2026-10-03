"""Wave-467 set-theory-5 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.constructible_l import bench_constructible_l
from quant_fund.models.core_model import bench_core_model
from quant_fund.models.large_card import bench_large_card
from quant_fund.models.pcf_theory import bench_pcf_theory
from quant_fund.models.proper_forcing import bench_proper_forcing
from quant_fund.models.square_princ import bench_square_princ

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


def bench_constructible_l_family(seed: int = _SEED + 2708) -> dict[str, float]:
    return _floats(_finite_blob("constructible_l", bench_constructible_l(seed)))


def bench_large_card_family(seed: int = _SEED + 2709) -> dict[str, float]:
    return _floats(_finite_blob("large_card", bench_large_card(seed)))


def bench_pcf_theory_family(seed: int = _SEED + 2710) -> dict[str, float]:
    return _floats(_finite_blob("pcf_theory", bench_pcf_theory(seed)))


def bench_proper_forcing_family(seed: int = _SEED + 2711) -> dict[str, float]:
    return _floats(_finite_blob("proper_forcing", bench_proper_forcing(seed)))


def bench_core_model_family(seed: int = _SEED + 2712) -> dict[str, float]:
    return _floats(_finite_blob("core_model", bench_core_model(seed)))


def bench_square_princ_family(seed: int = _SEED + 2713) -> dict[str, float]:
    return _floats(_finite_blob("square_princ", bench_square_princ(seed)))
