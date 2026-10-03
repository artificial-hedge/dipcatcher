"""Wave-716 triangulated-geometry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.exceptional_coll import (
    bench_exceptional_coll,
)
from quant_fund.models.fourier_mukai import (
    bench_fourier_mukai,
)
from quant_fund.models.semi_orthogonal import (
    bench_semi_orthogonal,
)
from quant_fund.models.serre_functor import (
    bench_serre_functor,
)
from quant_fund.models.sod_decomp import bench_sod_decomp
from quant_fund.models.spherical_functor import (
    bench_spherical_functor,
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


def bench_exceptional_coll_family(
    seed: int = _SEED + 10500,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "exceptional_coll",
            bench_exceptional_coll(seed),
        )
    )


def bench_spherical_functor_family(
    seed: int = _SEED + 10501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spherical_functor",
            bench_spherical_functor(seed),
        )
    )


def bench_serre_functor_family(
    seed: int = _SEED + 10502,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "serre_functor",
            bench_serre_functor(seed),
        )
    )


def bench_sod_decomp_family(
    seed: int = _SEED + 10503,
) -> dict[str, float]:
    return _floats(_finite_blob("sod_decomp", bench_sod_decomp(seed)))


def bench_fourier_mukai_family(
    seed: int = _SEED + 10504,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fourier_mukai",
            bench_fourier_mukai(seed),
        )
    )


def bench_semi_orthogonal_family(
    seed: int = _SEED + 10505,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "semi_orthogonal",
            bench_semi_orthogonal(seed),
        )
    )
