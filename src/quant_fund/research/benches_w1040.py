"""Wave-1040 robotics-engineering canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.actuator_design import bench_actuator_design
from quant_fund.models.motion_control import bench_motion_control
from quant_fund.models.path_planning import bench_path_planning
from quant_fund.models.robot_dynamics import bench_robot_dynamics
from quant_fund.models.robot_kinematics import bench_robot_kinematics
from quant_fund.models.sensor_fusion import bench_sensor_fusion

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in blob.items():
        if key.lower() in _FORBIDDEN:
            raise ValueError(f"forbidden metric key: {key}")
        if not math.isfinite(val):
            raise ValueError(f"non-finite metric: {key}")
        out[key] = float(val)
    return out


def _floats(xs: Iterable[float]) -> list[float]:
    return [float(x) for x in xs]


def bench_robot_kinematics_family(seed: int = _SEED + 42800) -> dict[str, float]:
    return _finite_blob(bench_robot_kinematics(seed))


def bench_robot_dynamics_family(seed: int = _SEED + 42801) -> dict[str, float]:
    return _finite_blob(bench_robot_dynamics(seed))


def bench_motion_control_family(seed: int = _SEED + 42802) -> dict[str, float]:
    return _finite_blob(bench_motion_control(seed))


def bench_sensor_fusion_family(seed: int = _SEED + 42803) -> dict[str, float]:
    return _finite_blob(bench_sensor_fusion(seed))


def bench_path_planning_family(seed: int = _SEED + 42804) -> dict[str, float]:
    return _finite_blob(bench_path_planning(seed))


def bench_actuator_design_family(seed: int = _SEED + 42805) -> dict[str, float]:
    return _finite_blob(bench_actuator_design(seed))
