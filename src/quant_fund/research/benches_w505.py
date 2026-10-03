"""Wave-505 cobordism-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cobordism_grp import bench_cobordism_grp
from quant_fund.models.complex_cob import bench_complex_cob
from quant_fund.models.framed_cob import bench_framed_cob
from quant_fund.models.oriented_cob import bench_oriented_cob
from quant_fund.models.thom_cob import bench_thom_cob
from quant_fund.models.unoriented_cob import bench_unoriented_cob

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


def bench_cobordism_grp_family(seed: int = _SEED + 2936) -> dict[str, float]:
    return _floats(_finite_blob("cobordism_grp", bench_cobordism_grp(seed)))


def bench_oriented_cob_family(seed: int = _SEED + 2937) -> dict[str, float]:
    return _floats(_finite_blob("oriented_cob", bench_oriented_cob(seed)))


def bench_unoriented_cob_family(seed: int = _SEED + 2938) -> dict[str, float]:
    return _floats(_finite_blob("unoriented_cob", bench_unoriented_cob(seed)))


def bench_complex_cob_family(seed: int = _SEED + 2939) -> dict[str, float]:
    return _floats(_finite_blob("complex_cob", bench_complex_cob(seed)))


def bench_framed_cob_family(seed: int = _SEED + 2940) -> dict[str, float]:
    return _floats(_finite_blob("framed_cob", bench_framed_cob(seed)))


def bench_thom_cob_family(seed: int = _SEED + 2941) -> dict[str, float]:
    return _floats(_finite_blob("thom_cob", bench_thom_cob(seed)))
