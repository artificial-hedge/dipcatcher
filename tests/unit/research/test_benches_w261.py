"""Wave-261 adapter tests."""

from quant_fund.research.benches_w261 import (
    bench_ekf_slam_family,
    bench_frontier_explore_family,
    bench_occupancy_grid_family,
    bench_particle_slam_family,
    bench_pure_pursuit_family,
    bench_stanley_family,
)


def test_bench_ekf_slam_family():
    out = bench_ekf_slam_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_occupancy_grid_family():
    out = bench_occupancy_grid_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_pure_pursuit_family():
    out = bench_pure_pursuit_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_stanley_family():
    out = bench_stanley_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_particle_slam_family():
    out = bench_particle_slam_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_frontier_explore_family():
    out = bench_frontier_explore_family()
    assert out and all(k.startswith("synthetic_") for k in out)
