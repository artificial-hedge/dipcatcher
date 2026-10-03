"""Wave-262 HPC unit tests."""

from quant_fund.models.mesi_cache import bench_mesi_cache
from quant_fund.models.numa_alloc import bench_numa_alloc
from quant_fund.models.ring_allreduce import bench_ring_allreduce
from quant_fund.models.simd_lanes import bench_simd_lanes
from quant_fund.models.stencil_halo import bench_stencil_halo
from quant_fund.models.task_dag import bench_task_dag


def test_stencil_keys() -> None:
    out = bench_stencil_halo(1)
    assert "synthetic_halo_max_err" in out


def test_mesi_keys() -> None:
    assert "synthetic_mesi_invariant" in bench_mesi_cache(2)


def test_ring_keys() -> None:
    assert "synthetic_ring_correct" in bench_ring_allreduce(3)


def test_simd_keys() -> None:
    assert "synthetic_simd_correct" in bench_simd_lanes(4)


def test_dag_keys() -> None:
    assert "synthetic_dag_scheduled" in bench_task_dag(5)


def test_numa_keys() -> None:
    assert "synthetic_numa_locality" in bench_numa_alloc(6)


def test_ranges() -> None:
    assert 0.0 <= bench_mesi_cache(7)["synthetic_mesi_invariant"] <= 1.0
    assert 0.0 <= bench_ring_allreduce(8)["synthetic_ring_correct"] <= 1.0


def test_determinism() -> None:
    assert bench_stencil_halo(9) == bench_stencil_halo(9)
