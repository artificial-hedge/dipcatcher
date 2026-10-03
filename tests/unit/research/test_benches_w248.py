"""Wave-248 adapter tests."""

from quant_fund.research.benches_w248 import (
    bench_brzozowski_deriv_family,
    bench_cellular_automata_family,
    bench_dfa_equiv_family,
    bench_mealy_moore_family,
    bench_pda_sim_family,
    bench_turing_machine_family,
)


def test_bench_brzozowski_deriv_family():
    out = bench_brzozowski_deriv_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_cellular_automata_family():
    out = bench_cellular_automata_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_dfa_equiv_family():
    out = bench_dfa_equiv_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_mealy_moore_family():
    out = bench_mealy_moore_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_pda_sim_family():
    out = bench_pda_sim_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_turing_machine_family():
    out = bench_turing_machine_family()
    assert out and all(k.startswith("synthetic_") for k in out)
