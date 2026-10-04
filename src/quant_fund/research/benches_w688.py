"""Wave-688 spectral-AG-7 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.spectral_cellular import (
    bench_spectral_cellular,
)
from quant_fund.models.spectral_cohomological import (
    bench_spectral_cohomological,
)
from quant_fund.models.spectral_field import bench_spectral_field
from quant_fund.models.spectral_filtration import (
    bench_spectral_filtration,
)
from quant_fund.models.spectral_finite import (
    bench_spectral_finite,
)
from quant_fund.models.spectral_lattice import (
    bench_spectral_lattice,
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


def bench_spectral_field_family(
    seed: int = _SEED + 7700,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_field", bench_spectral_field(seed)))


def bench_spectral_lattice_family(
    seed: int = _SEED + 7701,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_lattice", bench_spectral_lattice(seed)))


def bench_spectral_filtration_family(
    seed: int = _SEED + 7702,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_filtration",
            bench_spectral_filtration(seed),
        )
    )


def bench_spectral_cellular_family(
    seed: int = _SEED + 7703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_cellular",
            bench_spectral_cellular(seed),
        )
    )


def bench_spectral_cohomological_family(
    seed: int = _SEED + 7704,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_cohomological",
            bench_spectral_cohomological(seed),
        )
    )


def bench_spectral_finite_family(
    seed: int = _SEED + 7705,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_finite", bench_spectral_finite(seed)))
