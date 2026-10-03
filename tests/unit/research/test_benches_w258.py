"""Wave-258 adapter tests."""

from quant_fund.research.benches_w258 import (
    bench_adaptive_qp_family,
    bench_bitmap_index_family,
    bench_cascades_opt_family,
    bench_func_dep_family,
    bench_vectorized_exec_family,
    bench_zone_map_family,
)


def test_bench_cascades_opt_family():
    out = bench_cascades_opt_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_vectorized_exec_family():
    out = bench_vectorized_exec_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_zone_map_family():
    out = bench_zone_map_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_func_dep_family():
    out = bench_func_dep_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_bitmap_index_family():
    out = bench_bitmap_index_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_adaptive_qp_family():
    out = bench_adaptive_qp_family()
    assert out and all(k.startswith("synthetic_") for k in out)
