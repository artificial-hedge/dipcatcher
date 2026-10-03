"""Wave-841 orthogonal-polynomial bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chebyshev_t import (
    bench_chebyshev_t,
)
from quant_fund.models.gegenbauer_poly import (
    bench_gegenbauer_poly,
)
from quant_fund.models.hermite_poly import (
    bench_hermite_poly,
)
from quant_fund.models.jacobi_poly import (
    bench_jacobi_poly,
)
from quant_fund.models.laguerre_poly import (
    bench_laguerre_poly,
)
from quant_fund.models.legendre_poly import (
    bench_legendre_poly,
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


def bench_legendre_poly_family(
    seed: int = _SEED + 22900,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "legendre_poly",
            bench_legendre_poly(seed),
        )
    )


def bench_chebyshev_t_family(
    seed: int = _SEED + 22901,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "chebyshev_t",
            bench_chebyshev_t(seed),
        )
    )


def bench_hermite_poly_family(
    seed: int = _SEED + 22902,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hermite_poly",
            bench_hermite_poly(seed),
        )
    )


def bench_laguerre_poly_family(
    seed: int = _SEED + 22903,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "laguerre_poly",
            bench_laguerre_poly(seed),
        )
    )


def bench_jacobi_poly_family(
    seed: int = _SEED + 22904,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "jacobi_poly",
            bench_jacobi_poly(seed),
        )
    )


def bench_gegenbauer_poly_family(
    seed: int = _SEED + 22905,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gegenbauer_poly",
            bench_gegenbauer_poly(seed),
        )
    )
