"""Wave-842 spectral-methods bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chebyshev_collocation import (
    bench_chebyshev_collocation,
)
from quant_fund.models.chebyshev_grid import (
    bench_chebyshev_grid,
)
from quant_fund.models.dealiasing import (
    bench_dealiasing,
)
from quant_fund.models.fourier_galerkin import (
    bench_fourier_galerkin,
)
from quant_fund.models.legendre_tau import (
    bench_legendre_tau,
)
from quant_fund.models.spectral_deriv import (
    bench_spectral_deriv,
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


def bench_chebyshev_grid_family(
    seed: int = _SEED + 23000,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "chebyshev_grid",
            bench_chebyshev_grid(seed),
        )
    )


def bench_fourier_galerkin_family(
    seed: int = _SEED + 23001,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fourier_galerkin",
            bench_fourier_galerkin(seed),
        )
    )


def bench_legendre_tau_family(
    seed: int = _SEED + 23002,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "legendre_tau",
            bench_legendre_tau(seed),
        )
    )


def bench_chebyshev_collocation_family(
    seed: int = _SEED + 23003,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "chebyshev_collocation",
            bench_chebyshev_collocation(seed),
        )
    )


def bench_spectral_deriv_family(
    seed: int = _SEED + 23004,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_deriv",
            bench_spectral_deriv(seed),
        )
    )


def bench_dealiasing_family(
    seed: int = _SEED + 23005,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dealiasing",
            bench_dealiasing(seed),
        )
    )
