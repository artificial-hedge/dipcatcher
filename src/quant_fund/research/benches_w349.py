"""Wave-349 algebraic-geometry-3 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bezout_bezout import bench_bezout_bezout
from quant_fund.models.hilbert_poly import bench_hilbert_poly
from quant_fund.models.monomial_ideal import bench_monomial_ideal
from quant_fund.models.projective_plane import bench_projective_plane
from quant_fund.models.variety_dim import bench_variety_dim
from quant_fund.models.zariski_topo import bench_zariski_topo

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


def bench_zariski_topo_family(seed: int = _SEED + 2001) -> dict[str, float]:
    return _floats(_finite_blob("zariski_topo", bench_zariski_topo(seed)))


def bench_projective_plane_family(seed: int = _SEED + 2002) -> dict[str, float]:
    return _floats(_finite_blob("projective_plane", bench_projective_plane(seed)))


def bench_bezout_bezout_family(seed: int = _SEED + 2003) -> dict[str, float]:
    return _floats(_finite_blob("bezout_bezout", bench_bezout_bezout(seed)))


def bench_variety_dim_family(seed: int = _SEED + 2004) -> dict[str, float]:
    return _floats(_finite_blob("variety_dim", bench_variety_dim(seed)))


def bench_monomial_ideal_family(seed: int = _SEED + 2005) -> dict[str, float]:
    return _floats(_finite_blob("monomial_ideal", bench_monomial_ideal(seed)))


def bench_hilbert_poly_family(seed: int = _SEED + 2006) -> dict[str, float]:
    return _floats(_finite_blob("hilbert_poly", bench_hilbert_poly(seed)))
