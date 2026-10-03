"""Wave-699 homotopy-29 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.homotopy_general import (
    bench_homotopy_general,
)
from quant_fund.models.homotopy_rational import (
    bench_homotopy_rational,
)
from quant_fund.models.stable_dual import bench_stable_dual
from quant_fund.models.stable_lie import bench_stable_lie
from quant_fund.models.stable_motivic import (
    bench_stable_motivic,
)
from quant_fund.models.stable_perf import bench_stable_perf

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


def bench_homotopy_general_family(
    seed: int = _SEED + 8800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "homotopy_general",
            bench_homotopy_general(seed),
        )
    )


def bench_homotopy_rational_family(
    seed: int = _SEED + 8801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "homotopy_rational",
            bench_homotopy_rational(seed),
        )
    )


def bench_stable_dual_family(
    seed: int = _SEED + 8802,
) -> dict[str, float]:
    return _floats(_finite_blob("stable_dual", bench_stable_dual(seed)))


def bench_stable_lie_family(
    seed: int = _SEED + 8803,
) -> dict[str, float]:
    return _floats(_finite_blob("stable_lie", bench_stable_lie(seed)))


def bench_stable_motivic_family(
    seed: int = _SEED + 8804,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stable_motivic",
            bench_stable_motivic(seed),
        )
    )


def bench_stable_perf_family(
    seed: int = _SEED + 8805,
) -> dict[str, float]:
    return _floats(_finite_blob("stable_perf", bench_stable_perf(seed)))
