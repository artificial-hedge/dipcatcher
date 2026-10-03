"""Wave-252 adapter tests."""

from quant_fund.research.benches_w252 import (
    bench_anf_cps_family,
    bench_compacting_gc_family,
    bench_dispatch_table_family,
    bench_gen_gc_family,
    bench_poly_inline_cache_family,
    bench_trampoline_tc_family,
)


def test_bench_gen_gc_family():
    out = bench_gen_gc_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_compacting_gc_family():
    out = bench_compacting_gc_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_dispatch_table_family():
    out = bench_dispatch_table_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_poly_inline_cache_family():
    out = bench_poly_inline_cache_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_anf_cps_family():
    out = bench_anf_cps_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_trampoline_tc_family():
    out = bench_trampoline_tc_family()
    assert out and all(k.startswith("synthetic_") for k in out)
