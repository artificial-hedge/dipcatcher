"""Wave-262 adapter tests."""

from quant_fund.research.benches_w262 import (
    bench_mesi_cache_family,
    bench_numa_alloc_family,
    bench_ring_allreduce_family,
    bench_simd_lanes_family,
    bench_stencil_halo_family,
    bench_task_dag_family,
)


def test_bench_stencil_halo_family():
    assert all(k.startswith("synthetic_") for k in bench_stencil_halo_family())


def test_bench_mesi_cache_family():
    assert all(k.startswith("synthetic_") for k in bench_mesi_cache_family())


def test_bench_ring_allreduce_family():
    assert all(k.startswith("synthetic_") for k in bench_ring_allreduce_family())


def test_bench_simd_lanes_family():
    assert all(k.startswith("synthetic_") for k in bench_simd_lanes_family())


def test_bench_task_dag_family():
    assert all(k.startswith("synthetic_") for k in bench_task_dag_family())


def test_bench_numa_alloc_family():
    assert all(k.startswith("synthetic_") for k in bench_numa_alloc_family())
