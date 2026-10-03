"""Wave-614 homotopy-13 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.finite_spectra import bench_finite_spectra
from quant_fund.models.moore_spec import bench_moore_spec
from quant_fund.models.peterson_stein import bench_peterson_stein
from quant_fund.models.primary_op import bench_primary_op
from quant_fund.models.secondary_op import bench_secondary_op
from quant_fund.models.steenrod_sq import bench_steenrod_sq

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


def bench_primary_op_family(seed: int = _SEED + 3590) -> dict[str, float]:
    return _floats(_finite_blob("primary_op", bench_primary_op(seed)))


def bench_secondary_op_family(
    seed: int = _SEED + 3591,
) -> dict[str, float]:
    return _floats(_finite_blob("secondary_op", bench_secondary_op(seed)))


def bench_steenrod_sq_family(
    seed: int = _SEED + 3592,
) -> dict[str, float]:
    return _floats(_finite_blob("steenrod_sq", bench_steenrod_sq(seed)))


def bench_peterson_stein_family(
    seed: int = _SEED + 3593,
) -> dict[str, float]:
    return _floats(_finite_blob("peterson_stein", bench_peterson_stein(seed)))


def bench_moore_spec_family(seed: int = _SEED + 3594) -> dict[str, float]:
    return _floats(_finite_blob("moore_spec", bench_moore_spec(seed)))


def bench_finite_spectra_family(
    seed: int = _SEED + 3595,
) -> dict[str, float]:
    return _floats(_finite_blob("finite_spectra", bench_finite_spectra(seed)))
