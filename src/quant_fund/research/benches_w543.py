"""Wave-543 Riemann-surfaces bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.abel_jacobi import bench_abel_jacobi
from quant_fund.models.branched_cover import bench_branched_cover
from quant_fund.models.fuchsian_group import bench_fuchsian_group
from quant_fund.models.riemann_hurwitz import bench_riemann_hurwitz
from quant_fund.models.riemann_surface import bench_riemann_surface
from quant_fund.models.teichmuller_space import bench_teichmuller_space

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


def bench_riemann_surface_family(
    seed: int = _SEED + 3164,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "riemann_surface",
            bench_riemann_surface(seed),
        )
    )


def bench_branched_cover_family(
    seed: int = _SEED + 3165,
) -> dict[str, float]:
    return _floats(_finite_blob("branched_cover", bench_branched_cover(seed)))


def bench_abel_jacobi_family(seed: int = _SEED + 3166) -> dict[str, float]:
    return _floats(_finite_blob("abel_jacobi", bench_abel_jacobi(seed)))


def bench_riemann_hurwitz_family(
    seed: int = _SEED + 3167,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "riemann_hurwitz",
            bench_riemann_hurwitz(seed),
        )
    )


def bench_fuchsian_group_family(
    seed: int = _SEED + 3168,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fuchsian_group",
            bench_fuchsian_group(seed),
        )
    )


def bench_teichmuller_space_family(
    seed: int = _SEED + 3169,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "teichmuller_space",
            bench_teichmuller_space(seed),
        )
    )
