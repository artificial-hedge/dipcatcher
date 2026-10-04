"""Wave-1040 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1040 import (
    bench_actuator_design_family,
    bench_motion_control_family,
    bench_path_planning_family,
    bench_robot_dynamics_family,
    bench_robot_kinematics_family,
    bench_sensor_fusion_family,
)

_FAMILY_BENCHES = [
    bench_robot_kinematics_family,
    bench_robot_dynamics_family,
    bench_motion_control_family,
    bench_sensor_fusion_family,
    bench_path_planning_family,
    bench_actuator_design_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
