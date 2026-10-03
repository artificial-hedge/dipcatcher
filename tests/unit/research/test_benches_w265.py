"""Wave-265 adapter tests."""

from quant_fund.research.benches_w265 import (
    bench_asan_shadow_family,
    bench_contract_check_family,
    bench_fuzzer_mutate_family,
    bench_grammar_fuzz_family,
    bench_symbolic_exec_family,
    bench_taint_track_family,
)


def test_bench_fuzzer_mutate_family():
    assert all(k.startswith("synthetic_") for k in bench_fuzzer_mutate_family())


def test_bench_taint_track_family():
    assert all(k.startswith("synthetic_") for k in bench_taint_track_family())


def test_bench_asan_shadow_family():
    assert all(k.startswith("synthetic_") for k in bench_asan_shadow_family())


def test_bench_symbolic_exec_family():
    assert all(k.startswith("synthetic_") for k in bench_symbolic_exec_family())


def test_bench_contract_check_family():
    assert all(k.startswith("synthetic_") for k in bench_contract_check_family())


def test_bench_grammar_fuzz_family():
    assert all(k.startswith("synthetic_") for k in bench_grammar_fuzz_family())
