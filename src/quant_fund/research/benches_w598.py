"""Wave-598 algebraic-K-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bloch_beilinson import (
    bench_bloch_beilinson,
)
from quant_fund.models.borel_regulator import (
    bench_borel_regulator,
)
from quant_fund.models.etale_ktheory import (
    bench_etale_ktheory,
)
from quant_fund.models.lichtenbaum_k import (
    bench_lichtenbaum_k,
)
from quant_fund.models.soul_elem import bench_soul_elem
from quant_fund.models.thh_trace import bench_thh_trace

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


def bench_borel_regulator_family(
    seed: int = _SEED + 3494,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "borel_regulator",
            bench_borel_regulator(seed),
        )
    )


def bench_soul_elem_family(
    seed: int = _SEED + 3495,
) -> dict[str, float]:
    return _floats(_finite_blob("soul_elem", bench_soul_elem(seed)))


def bench_lichtenbaum_k_family(
    seed: int = _SEED + 3496,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "lichtenbaum_k",
            bench_lichtenbaum_k(seed),
        )
    )


def bench_bloch_beilinson_family(
    seed: int = _SEED + 3497,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bloch_beilinson",
            bench_bloch_beilinson(seed),
        )
    )


def bench_etale_ktheory_family(
    seed: int = _SEED + 3498,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "etale_ktheory",
            bench_etale_ktheory(seed),
        )
    )


def bench_thh_trace_family(seed: int = _SEED + 3499) -> dict[str, float]:
    return _floats(_finite_blob("thh_trace", bench_thh_trace(seed)))
