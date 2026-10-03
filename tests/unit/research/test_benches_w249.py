"""Wave-249 adapter tests."""

from quant_fund.research.benches_w249 import (
    bench_debruijn_assemble_family,
    bench_fm_index_family,
    bench_motif_scan_family,
    bench_needleman_wunsch_family,
    bench_smith_waterman_family,
    bench_upgma_tree_family,
)


def test_bench_debruijn_assemble_family():
    out = bench_debruijn_assemble_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_fm_index_family():
    out = bench_fm_index_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_motif_scan_family():
    out = bench_motif_scan_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_needleman_wunsch_family():
    out = bench_needleman_wunsch_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_smith_waterman_family():
    out = bench_smith_waterman_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_upgma_tree_family():
    out = bench_upgma_tree_family()
    assert out and all(k.startswith("synthetic_") for k in out)
