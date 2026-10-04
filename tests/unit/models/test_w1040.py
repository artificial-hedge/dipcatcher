"""Wave-1040 robotics-engineering canon tests."""

from __future__ import annotations

from quant_fund.models.actuator_design import bench_actuator_design
from quant_fund.models.motion_control import bench_motion_control
from quant_fund.models.path_planning import bench_path_planning
from quant_fund.models.robot_dynamics import bench_robot_dynamics
from quant_fund.models.robot_kinematics import bench_robot_kinematics
from quant_fund.models.sensor_fusion import bench_sensor_fusion


def test_robot_kinematics():
    assert bench_robot_kinematics()["synthetic_robot_kinematics"] == 1.0


def test_robot_dynamics():
    assert bench_robot_dynamics()["synthetic_robot_dynamics"] == 1.0


def test_motion_control():
    assert bench_motion_control()["synthetic_motion_control"] == 1.0


def test_sensor_fusion():
    assert bench_sensor_fusion()["synthetic_sensor_fusion"] == 1.0


def test_path_planning():
    assert bench_path_planning()["synthetic_path_planning"] == 1.0


def test_actuator_design():
    assert bench_actuator_design()["synthetic_actuator_design"] == 1.0
