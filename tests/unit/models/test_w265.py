"""Wave-265 program-analysis unit tests."""

from quant_fund.models.asan_shadow import bench_asan_shadow
from quant_fund.models.contract_check import bench_contract_check
from quant_fund.models.fuzzer_mutate import bench_fuzzer_mutate
from quant_fund.models.grammar_fuzz import bench_grammar_fuzz
from quant_fund.models.symbolic_exec import bench_symbolic_exec
from quant_fund.models.taint_track import bench_taint_track


def test_fuzzer_keys() -> None:
    assert "synthetic_fuzz_coverage" in bench_fuzzer_mutate(1)


def test_taint_keys() -> None:
    assert "synthetic_taint_correct" in bench_taint_track(2)


def test_asan_keys() -> None:
    assert "synthetic_asan_detects" in bench_asan_shadow(3)


def test_sym_keys() -> None:
    assert "synthetic_sym_paths" in bench_symbolic_exec(4)


def test_contract_keys() -> None:
    assert "synthetic_contract_holds" in bench_contract_check(5)


def test_grammar_keys() -> None:
    assert "synthetic_grammar_valid" in bench_grammar_fuzz(6)


def test_ranges() -> None:
    assert 0.0 <= bench_taint_track(7)["synthetic_taint_correct"] <= 1.0
    assert 0.0 <= bench_grammar_fuzz(8)["synthetic_grammar_valid"] <= 1.0


def test_determinism() -> None:
    assert bench_fuzzer_mutate(9) == bench_fuzzer_mutate(9)
