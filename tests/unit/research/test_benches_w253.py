"""Wave-253 adapter tests."""

from quant_fund.research.benches_w253 import (
    bench_buchi_automata_family,
    bench_cfg_pda_equiv_family,
    bench_register_automata_family,
    bench_tree_automata_family,
    bench_two_way_dfa_family,
    bench_weighted_fst_family,
)


def test_bench_tree_automata_family():
    out = bench_tree_automata_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_buchi_automata_family():
    out = bench_buchi_automata_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_weighted_fst_family():
    out = bench_weighted_fst_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_cfg_pda_equiv_family():
    out = bench_cfg_pda_equiv_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_two_way_dfa_family():
    out = bench_two_way_dfa_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_register_automata_family():
    out = bench_register_automata_family()
    assert out and all(k.startswith("synthetic_") for k in out)
