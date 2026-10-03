"""Wave-244 adapter tests."""

from quant_fund.research.benches_w244 import (
    bench_inverted_index_family,
    bench_lsh_dedup_family,
    bench_ngram_spell_family,
    bench_positional_index_family,
    bench_posting_merge_family,
    bench_wand_bmw_family,
)


def test_bench_inverted_index_family():
    out = bench_inverted_index_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_lsh_dedup_family():
    out = bench_lsh_dedup_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_ngram_spell_family():
    out = bench_ngram_spell_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_positional_index_family():
    out = bench_positional_index_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_posting_merge_family():
    out = bench_posting_merge_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_wand_bmw_family():
    out = bench_wand_bmw_family()
    assert out and all(k.startswith("synthetic_") for k in out)
