"""Wave-450 stable-infinity bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.exact_seq import bench_exact_seq
from quant_fund.models.smash_monoidal import bench_smash_monoidal
from quant_fund.models.spectra_cat import bench_spectra_cat
from quant_fund.models.stable_infty import bench_stable_infty
from quant_fund.models.stable_tstruct import bench_stable_tstruct
from quant_fund.models.thh_tc import bench_thh_tc

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


def bench_stable_infty_family(seed: int = _SEED + 2606) -> dict[str, float]:
    return _floats(_finite_blob("stable_infty", bench_stable_infty(seed)))


def bench_spectra_cat_family(seed: int = _SEED + 2607) -> dict[str, float]:
    return _floats(_finite_blob("spectra_cat", bench_spectra_cat(seed)))


def bench_exact_seq_family(seed: int = _SEED + 2608) -> dict[str, float]:
    return _floats(_finite_blob("exact_seq", bench_exact_seq(seed)))


def bench_stable_tstruct_family(seed: int = _SEED + 2609) -> dict[str, float]:
    return _floats(_finite_blob("t_structure", bench_stable_tstruct(seed)))


def bench_smash_monoidal_family(seed: int = _SEED + 2610) -> dict[str, float]:
    return _floats(_finite_blob("smash_monoidal", bench_smash_monoidal(seed)))


def bench_thh_tc_family(seed: int = _SEED + 2611) -> dict[str, float]:
    return _floats(_finite_blob("thh_tc", bench_thh_tc(seed)))
