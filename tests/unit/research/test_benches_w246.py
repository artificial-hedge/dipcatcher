"""Wave-246 adapter tests."""

from quant_fund.research.benches_w246 import (
    bench_bytecode_vm_family,
    bench_closure_conv_family,
    bench_inline_cache_family,
    bench_nan_tagging_family,
    bench_tail_call_tramp_family,
    bench_threaded_interp_family,
)


def test_bench_bytecode_vm_family():
    out = bench_bytecode_vm_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_closure_conv_family():
    out = bench_closure_conv_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_inline_cache_family():
    out = bench_inline_cache_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_nan_tagging_family():
    out = bench_nan_tagging_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_tail_call_tramp_family():
    out = bench_tail_call_tramp_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_threaded_interp_family():
    out = bench_threaded_interp_family()
    assert out and all(k.startswith("synthetic_") for k in out)
