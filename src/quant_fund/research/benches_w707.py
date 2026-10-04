"""Wave-707 spectral-AG-9 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.spectral_coord import bench_spectral_coord
from quant_fund.models.spectral_ext_field import (
    bench_spectral_ext_field,
)
from quant_fund.models.spectral_level import bench_spectral_level
from quant_fund.models.spectral_polynomial2 import (
    bench_spectral_polynomial2,
)
from quant_fund.models.spectral_prime import bench_spectral_prime
from quant_fund.models.spectral_residue import (
    bench_spectral_residue,
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


def bench_spectral_prime_family(
    seed: int = _SEED + 9600,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_prime", bench_spectral_prime(seed)))


def bench_spectral_residue_family(
    seed: int = _SEED + 9601,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_residue",
            bench_spectral_residue(seed),
        )
    )


def bench_spectral_level_family(
    seed: int = _SEED + 9602,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_level", bench_spectral_level(seed)))


def bench_spectral_polynomial2_family(
    seed: int = _SEED + 9603,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_polynomial2",
            bench_spectral_polynomial2(seed),
        )
    )


def bench_spectral_coord_family(
    seed: int = _SEED + 9604,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_coord", bench_spectral_coord(seed)))


def bench_spectral_ext_field_family(
    seed: int = _SEED + 9605,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_ext_field",
            bench_spectral_ext_field(seed),
        )
    )
