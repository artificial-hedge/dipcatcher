"""Wave-693 derived-geometry-8 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.derived_cartesian import (
    bench_derived_cartesian,
)
from quant_fund.models.derived_etale import (
    bench_derived_etale,
)
from quant_fund.models.derived_flat import bench_derived_flat
from quant_fund.models.derived_quasi_coherent import (
    bench_derived_quasi_coherent,
)
from quant_fund.models.derived_represent import (
    bench_derived_represent,
)
from quant_fund.models.derived_smooth2 import (
    bench_derived_smooth2,
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


def bench_derived_etale_family(
    seed: int = _SEED + 8200,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_etale", bench_derived_etale(seed)))


def bench_derived_flat_family(
    seed: int = _SEED + 8201,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_flat", bench_derived_flat(seed)))


def bench_derived_smooth2_family(
    seed: int = _SEED + 8202,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_smooth2", bench_derived_smooth2(seed)))


def bench_derived_quasi_coherent_family(
    seed: int = _SEED + 8203,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "derived_quasi_coherent",
            bench_derived_quasi_coherent(seed),
        )
    )


def bench_derived_represent_family(
    seed: int = _SEED + 8204,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "derived_represent",
            bench_derived_represent(seed),
        )
    )


def bench_derived_cartesian_family(
    seed: int = _SEED + 8205,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "derived_cartesian",
            bench_derived_cartesian(seed),
        )
    )
