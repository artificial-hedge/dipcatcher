"""Wave-557 3-manifold-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dehn_surgery import bench_dehn_surgery
from quant_fund.models.heegaard_splitting import bench_heegaard_splitting
from quant_fund.models.normal_surface import bench_normal_surface
from quant_fund.models.sutured_mfd import bench_sutured_mfd
from quant_fund.models.taut_foliation import bench_taut_foliation
from quant_fund.models.thin_position import bench_thin_position

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


def bench_heegaard_splitting_family(
    seed: int = _SEED + 3248,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "heegaard_splitting",
            bench_heegaard_splitting(seed),
        )
    )


def bench_dehn_surgery_family(seed: int = _SEED + 3249) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dehn_surgery",
            bench_dehn_surgery(seed),
        )
    )


def bench_sutured_mfd_family(seed: int = _SEED + 3250) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "sutured_mfd",
            bench_sutured_mfd(seed),
        )
    )


def bench_taut_foliation_family(seed: int = _SEED + 3251) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "taut_foliation",
            bench_taut_foliation(seed),
        )
    )


def bench_thin_position_family(seed: int = _SEED + 3252) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "thin_position",
            bench_thin_position(seed),
        )
    )


def bench_normal_surface_family(seed: int = _SEED + 3253) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "normal_surface",
            bench_normal_surface(seed),
        )
    )
