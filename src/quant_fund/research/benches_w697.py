"""Wave-697 spectral-AG-8 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.spectral_dedekind import (
    bench_spectral_dedekind,
)
from quant_fund.models.spectral_dvr import bench_spectral_dvr
from quant_fund.models.spectral_excellent import (
    bench_spectral_excellent,
)
from quant_fund.models.spectral_jacobson import (
    bench_spectral_jacobson,
)
from quant_fund.models.spectral_noether import (
    bench_spectral_noether,
)
from quant_fund.models.spectral_regular import (
    bench_spectral_regular,
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


def bench_spectral_dvr_family(
    seed: int = _SEED + 8600,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_dvr", bench_spectral_dvr(seed)))


def bench_spectral_noether_family(
    seed: int = _SEED + 8601,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_noether", bench_spectral_noether(seed)))


def bench_spectral_regular_family(
    seed: int = _SEED + 8602,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_regular", bench_spectral_regular(seed)))


def bench_spectral_dedekind_family(
    seed: int = _SEED + 8603,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_dedekind",
            bench_spectral_dedekind(seed),
        )
    )


def bench_spectral_jacobson_family(
    seed: int = _SEED + 8604,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_jacobson",
            bench_spectral_jacobson(seed),
        )
    )


def bench_spectral_excellent_family(
    seed: int = _SEED + 8605,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_excellent",
            bench_spectral_excellent(seed),
        )
    )
