"""Wave-695 homotopy-28 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.homotopy_abelian import (
    bench_homotopy_abelian,
)
from quant_fund.models.homotopy_extended import (
    bench_homotopy_extended,
)
from quant_fund.models.homotopy_finite import (
    bench_homotopy_finite,
)
from quant_fund.models.homotopy_infinite import (
    bench_homotopy_infinite,
)
from quant_fund.models.stable_compact import (
    bench_stable_compact,
)
from quant_fund.models.stable_synthetic import (
    bench_stable_synthetic,
)

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


def bench_homotopy_abelian_family(
    seed: int = _SEED + 8400,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "homotopy_abelian",
            bench_homotopy_abelian(seed),
        )
    )


def bench_homotopy_finite_family(
    seed: int = _SEED + 8401,
) -> dict[str, float]:
    return _floats(_finite_blob("homotopy_finite", bench_homotopy_finite(seed)))


def bench_homotopy_infinite_family(
    seed: int = _SEED + 8402,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "homotopy_infinite",
            bench_homotopy_infinite(seed),
        )
    )


def bench_homotopy_extended_family(
    seed: int = _SEED + 8403,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "homotopy_extended",
            bench_homotopy_extended(seed),
        )
    )


def bench_stable_synthetic_family(
    seed: int = _SEED + 8404,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stable_synthetic",
            bench_stable_synthetic(seed),
        )
    )


def bench_stable_compact_family(
    seed: int = _SEED + 8405,
) -> dict[str, float]:
    return _floats(_finite_blob("stable_compact", bench_stable_compact(seed)))
