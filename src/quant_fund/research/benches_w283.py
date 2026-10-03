"""Wave-283 robotics-3 benches: kinematics, sensing, planning, fusion."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bezier_curve import bench_bezier_curve
from quant_fund.models.fk_dh import bench_fk_dh
from quant_fund.models.ik_jac import bench_ik_jac
from quant_fund.models.odom_comp import bench_odom_comp
from quant_fund.models.pot_field import bench_pot_field
from quant_fund.models.ray_lidar import bench_ray_lidar

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


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


def bench_fk_dh_family(seed: int = _SEED + 1600) -> dict[str, float]:
    return _floats(_finite_blob("fk_dh", bench_fk_dh(seed)))


def bench_ik_jac_family(seed: int = _SEED + 1601) -> dict[str, float]:
    return _floats(_finite_blob("ik_jac", bench_ik_jac(seed)))


def bench_ray_lidar_family(seed: int = _SEED + 1602) -> dict[str, float]:
    return _floats(_finite_blob("ray_lidar", bench_ray_lidar(seed)))


def bench_pot_field_family(seed: int = _SEED + 1603) -> dict[str, float]:
    return _floats(_finite_blob("pot_field", bench_pot_field(seed)))


def bench_bezier_curve_family(seed: int = _SEED + 1604) -> dict[str, float]:
    return _floats(_finite_blob("bezier_curve", bench_bezier_curve(seed)))


def bench_odom_comp_family(seed: int = _SEED + 1605) -> dict[str, float]:
    return _floats(_finite_blob("odom_comp", bench_odom_comp(seed)))
