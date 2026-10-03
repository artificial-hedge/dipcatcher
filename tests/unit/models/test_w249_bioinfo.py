"""Wave-249 bioinformatics canon tests."""

from quant_fund.models.debruijn_assemble import assemble, bench_debruijn_assemble
from quant_fund.models.fm_index import FMIndex, bench_fm_index
from quant_fund.models.motif_scan import bench_motif_scan, build_pssm, scan
from quant_fund.models.needleman_wunsch import bench_needleman_wunsch, nw_affine
from quant_fund.models.smith_waterman import bench_smith_waterman, smith_waterman
from quant_fund.models.upgma_tree import bench_upgma_tree, upgma


def test_sw_basic():
    s, aa, bb = smith_waterman("ACGT", "CGT")
    assert s > 0


def test_sw_bench():
    assert bench_smith_waterman()["synthetic_score_exact"] == 1.0


def test_nw_basic():
    s, aa, bb = nw_affine("ACG", "AG")
    assert aa.replace("-", "") == "ACG"


def test_nw_bench():
    assert bench_needleman_wunsch()["synthetic_score_exact"] == 1.0


def test_debruijn_basic():
    kms = ["ACG", "CGT", "GTA"]
    assert assemble(kms, 3) is not None


def test_debruijn_bench():
    assert bench_debruijn_assemble()["synthetic_kmers_covered"] == 1.0


def test_fm_basic():
    fm = FMIndex("ACGACG")
    assert fm.count("ACG") == 2


def test_fm_bench():
    assert bench_fm_index()["synthetic_count_exact"] == 1.0


def test_upgma_basic():
    d = {frozenset({"a", "b"}): 2.0, frozenset({"a", "c"}): 8.0, frozenset({"b", "c"}): 8.0}
    t = upgma(d, ["a", "b", "c"])
    assert t is not None


def test_upgma_bench():
    assert bench_upgma_tree()["synthetic_quartet_correct"] == 1.0


def test_motif_basic():
    pssm = build_pssm(["GATAAG", "GATAAG"])
    hits = scan("TTGATAAGTT", pssm)
    assert hits and max(hits, key=lambda x: x[1])[0] == 2


def test_motif_bench():
    assert bench_motif_scan()["synthetic_beats_control"] == 1.0
