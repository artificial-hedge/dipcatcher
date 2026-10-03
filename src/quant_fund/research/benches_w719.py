"""Wave-719 Calabi-Yau/Gorenstein bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.calabi_yau_tri import (
    bench_calabi_yau_tri,
)
from quant_fund.models.d_calabi_yau import bench_d_calabi_yau
from quant_fund.models.frobenius_cat import bench_frobenius_cat
from quant_fund.models.gorenstein_proj import (
    bench_gorenstein_proj,
)
from quant_fund.models.orbit_category import bench_orbit_category
from quant_fund.models.stable_category import (
    bench_stable_category,
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


def bench_calabi_yau_tri_family(
    seed: int = _SEED + 10800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "calabi_yau_tri",
            bench_calabi_yau_tri(seed),
        )
    )


def bench_d_calabi_yau_family(
    seed: int = _SEED + 10801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "d_calabi_yau",
            bench_d_calabi_yau(seed),
        )
    )


def bench_gorenstein_proj_family(
    seed: int = _SEED + 10802,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gorenstein_proj",
            bench_gorenstein_proj(seed),
        )
    )


def bench_frobenius_cat_family(
    seed: int = _SEED + 10803,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "frobenius_cat",
            bench_frobenius_cat(seed),
        )
    )


def bench_stable_category_family(
    seed: int = _SEED + 10804,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stable_category",
            bench_stable_category(seed),
        )
    )


def bench_orbit_category_family(
    seed: int = _SEED + 10805,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "orbit_category",
            bench_orbit_category(seed),
        )
    )
