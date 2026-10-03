"""Wave-857 classical-quadrature bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.clenshaw_curtis import (
    bench_clenshaw_curtis,
)
from quant_fund.models.fejer_quad import (
    bench_fejer_quad,
)
from quant_fund.models.gauss_chebyshev import (
    bench_gauss_chebyshev,
)
from quant_fund.models.gauss_kronrod import (
    bench_gauss_kronrod,
)
from quant_fund.models.gauss_legendre import (
    bench_gauss_legendre,
)
from quant_fund.models.newton_cotes import (
    bench_newton_cotes,
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


def bench_gauss_legendre_family(
    seed: int = _SEED + 24500,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gauss_legendre",
            bench_gauss_legendre(seed),
        )
    )


def bench_gauss_chebyshev_family(
    seed: int = _SEED + 24501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gauss_chebyshev",
            bench_gauss_chebyshev(seed),
        )
    )


def bench_clenshaw_curtis_family(
    seed: int = _SEED + 24502,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "clenshaw_curtis",
            bench_clenshaw_curtis(seed),
        )
    )


def bench_newton_cotes_family(
    seed: int = _SEED + 24503,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "newton_cotes",
            bench_newton_cotes(seed),
        )
    )


def bench_gauss_kronrod_family(
    seed: int = _SEED + 24504,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gauss_kronrod",
            bench_gauss_kronrod(seed),
        )
    )


def bench_fejer_quad_family(
    seed: int = _SEED + 24505,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fejer_quad",
            bench_fejer_quad(seed),
        )
    )
