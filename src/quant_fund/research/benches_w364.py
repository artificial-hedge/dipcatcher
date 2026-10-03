"""Wave-364 functional-analysis-3 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adjoint_op import bench_adjoint_op
from quant_fund.models.compact_resolvent import bench_compact_resolvent
from quant_fund.models.hahn_banach import bench_hahn_banach
from quant_fund.models.projection_thm import bench_projection_thm
from quant_fund.models.riesz_repr import bench_riesz_repr
from quant_fund.models.selfadjoint_spectrum import bench_selfadjoint_spectrum

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


def bench_hahn_banach_family(seed: int = _SEED + 2091) -> dict[str, float]:
    return _floats(_finite_blob("hahn_banach", bench_hahn_banach(seed)))


def bench_riesz_repr_family(seed: int = _SEED + 2092) -> dict[str, float]:
    return _floats(_finite_blob("riesz_repr", bench_riesz_repr(seed)))


def bench_adjoint_op_family(seed: int = _SEED + 2093) -> dict[str, float]:
    return _floats(_finite_blob("adjoint_op", bench_adjoint_op(seed)))


def bench_selfadjoint_spectrum_family(seed: int = _SEED + 2094) -> dict[str, float]:
    return _floats(_finite_blob("selfadjoint_spectrum", bench_selfadjoint_spectrum(seed)))


def bench_compact_resolvent_family(seed: int = _SEED + 2095) -> dict[str, float]:
    return _floats(_finite_blob("compact_resolvent", bench_compact_resolvent(seed)))


def bench_projection_thm_family(seed: int = _SEED + 2096) -> dict[str, float]:
    return _floats(_finite_blob("projection_thm", bench_projection_thm(seed)))
