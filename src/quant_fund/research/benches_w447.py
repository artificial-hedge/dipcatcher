"""Wave-447 TQFT bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bord_cat import bench_bord_cat
from quant_fund.models.chern_simons import bench_chern_simons
from quant_fund.models.dw_theory import bench_dw_theory
from quant_fund.models.extended_tqft import bench_extended_tqft
from quant_fund.models.frobenius_2d import bench_frobenius_2d
from quant_fund.models.tqft_axiom import bench_tqft_axiom

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


def bench_tqft_axiom_family(seed: int = _SEED + 2588) -> dict[str, float]:
    return _floats(_finite_blob("tqft_axiom", bench_tqft_axiom(seed)))


def bench_bord_cat_family(seed: int = _SEED + 2589) -> dict[str, float]:
    return _floats(_finite_blob("bord_cat", bench_bord_cat(seed)))


def bench_frobenius_2d_family(seed: int = _SEED + 2590) -> dict[str, float]:
    return _floats(_finite_blob("frobenius_2d", bench_frobenius_2d(seed)))


def bench_extended_tqft_family(seed: int = _SEED + 2591) -> dict[str, float]:
    return _floats(_finite_blob("extended_tqft", bench_extended_tqft(seed)))


def bench_dw_theory_family(seed: int = _SEED + 2592) -> dict[str, float]:
    return _floats(_finite_blob("dw_theory", bench_dw_theory(seed)))


def bench_chern_simons_family(seed: int = _SEED + 2593) -> dict[str, float]:
    return _floats(_finite_blob("chern_simons", bench_chern_simons(seed)))
