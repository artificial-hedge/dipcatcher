"""Wave-713 spectral-AG-10 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.derived_affine import (
    bench_derived_affine,
)
from quant_fund.models.derived_projective import (
    bench_derived_projective,
)
from quant_fund.models.spectral_artin import (
    bench_spectral_artin,
)
from quant_fund.models.spectral_dirac import (
    bench_spectral_dirac,
)
from quant_fund.models.spectral_gal import bench_spectral_gal
from quant_fund.models.spectral_semi import bench_spectral_semi

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


def bench_spectral_semi_family(
    seed: int = _SEED + 10200,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_semi", bench_spectral_semi(seed)))


def bench_spectral_artin_family(
    seed: int = _SEED + 10201,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_artin",
            bench_spectral_artin(seed),
        )
    )


def bench_spectral_gal_family(
    seed: int = _SEED + 10202,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_gal", bench_spectral_gal(seed)))


def bench_spectral_dirac_family(
    seed: int = _SEED + 10203,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_dirac",
            bench_spectral_dirac(seed),
        )
    )


def bench_derived_affine_family(
    seed: int = _SEED + 10204,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "derived_affine",
            bench_derived_affine(seed),
        )
    )


def bench_derived_projective_family(
    seed: int = _SEED + 10205,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "derived_projective",
            bench_derived_projective(seed),
        )
    )
