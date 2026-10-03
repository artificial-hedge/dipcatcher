"""Wave-261 robotics-2 unit tests."""

import numpy as np

from quant_fund.models.ekf_slam import bench_ekf_slam
from quant_fund.models.frontier_explore import bench_frontier_explore
from quant_fund.models.occupancy_grid import bench_occupancy_grid
from quant_fund.models.particle_slam import bench_particle_slam
from quant_fund.models.pure_pursuit import bench_pure_pursuit
from quant_fund.models.stanley import bench_stanley


def test_ekf_slam_keys() -> None:
    out = bench_ekf_slam(1)
    assert "synthetic_ekf_final_err" in out and "synthetic_ekf_beats_drift" in out


def test_ekf_slam_bounded() -> None:
    assert 0.0 <= bench_ekf_slam(3)["synthetic_ekf_beats_drift"] <= 1.0


def test_occupancy_keys() -> None:
    assert "synthetic_map_accuracy" in bench_occupancy_grid(2)


def test_occupancy_range() -> None:
    assert 0.0 <= bench_occupancy_grid(4)["synthetic_map_accuracy"] <= 1.0


def test_pure_pursuit_keys() -> None:
    assert "synthetic_pp_win" in bench_pure_pursuit(5)


def test_stanley_keys() -> None:
    assert "synthetic_stanley_cte" in bench_stanley(6)


def test_particle_slam_keys() -> None:
    assert "synthetic_slam_err" in bench_particle_slam(7)


def test_frontier_keys() -> None:
    assert "synthetic_frontier_coverage" in bench_frontier_explore(8)


def test_frontier_range() -> None:
    assert 0.0 <= bench_frontier_explore(9)["synthetic_frontier_coverage"] <= 1.0


def test_determinism() -> None:
    assert bench_ekf_slam(11) == bench_ekf_slam(11)
    assert isinstance(np.mean([1, 2]), float)
