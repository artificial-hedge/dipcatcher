"""Wave-888 RBF/basis bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.galerkin_projection import (
    bench_galerkin_projection,
)
from quant_fund.models.periodic_spline import (
    bench_periodic_spline,
)
from quant_fund.models.polyharmonic_rbf import (
    bench_polyharmonic_rbf,
)
from quant_fund.models.thin_plate_spline import (
    bench_thin_plate_spline,
)
from quant_fund.models.trefethen_diff import (
    bench_trefethen_diff,
)
from quant_fund.models.zernike_poly import (
    bench_zernike_poly,
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


def bench_thin_plate_spline_family(
    seed: int = _SEED + 27600,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "thin_plate_spline",
            bench_thin_plate_spline(seed),
        )
    )


def bench_polyharmonic_rbf_family(
    seed: int = _SEED + 27601,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "polyharmonic_rbf",
            bench_polyharmonic_rbf(seed),
        )
    )


def bench_trefethen_diff_family(
    seed: int = _SEED + 27602,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "trefethen_diff",
            bench_trefethen_diff(seed),
        )
    )


def bench_galerkin_projection_family(
    seed: int = _SEED + 27603,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "galerkin_projection",
            bench_galerkin_projection(seed),
        )
    )


def bench_periodic_spline_family(
    seed: int = _SEED + 27604,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "periodic_spline",
            bench_periodic_spline(seed),
        )
    )


def bench_zernike_poly_family(
    seed: int = _SEED + 27605,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "zernike_poly",
            bench_zernike_poly(seed),
        )
    )
