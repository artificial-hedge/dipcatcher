"""Wave-555 homological-mirror-symmetry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.hms_conjecture import bench_hms_conjecture
from quant_fund.models.landau_ginzburg import bench_landau_ginzburg
from quant_fund.models.mirror_functor import bench_mirror_functor
from quant_fund.models.syz_mirror import bench_syz_mirror
from quant_fund.models.torus_fibration import bench_torus_fibration
from quant_fund.models.wrapped_fukaya import bench_wrapped_fukaya

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


def bench_hms_conjecture_family(seed: int = _SEED + 3236) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hms_conjecture",
            bench_hms_conjecture(seed),
        )
    )


def bench_landau_ginzburg_family(
    seed: int = _SEED + 3237,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "landau_ginzburg",
            bench_landau_ginzburg(seed),
        )
    )


def bench_syz_mirror_family(seed: int = _SEED + 3238) -> dict[str, float]:
    return _floats(_finite_blob("syz_mirror", bench_syz_mirror(seed)))


def bench_torus_fibration_family(
    seed: int = _SEED + 3239,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "torus_fibration",
            bench_torus_fibration(seed),
        )
    )


def bench_wrapped_fukaya_family(seed: int = _SEED + 3240) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "wrapped_fukaya",
            bench_wrapped_fukaya(seed),
        )
    )


def bench_mirror_functor_family(seed: int = _SEED + 3241) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mirror_functor",
            bench_mirror_functor(seed),
        )
    )
