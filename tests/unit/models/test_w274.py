"""Wave-274 bioinformatics-2 module tests."""

import numpy as np

from quant_fund.models.gc_skew import gc_skew
from quant_fund.models.hmm_profile import build_profile, loglik
from quant_fund.models.kmer_count import kmer_spectrum
from quant_fund.models.orf_find import longest_orf
from quant_fund.models.seq_logo import column_ic
from quant_fund.models.star_msa import global_align


def test_profile_prefers_consensus() -> None:
    aln = ["ACGT", "ACGT", "ACGA"]
    prof = build_profile(aln)
    assert loglik("ACGT", prof) > loglik("TTTT", prof)


def test_profile_shape() -> None:
    prof = build_profile(["AC", "AC"])
    assert prof.shape == (2, 4)
    np.testing.assert_allclose(prof.sum(axis=1), 1.0)


def test_global_align_identical() -> None:
    assert global_align("ACGT", "ACGT") == 4


def test_global_align_gap() -> None:
    assert global_align("ACGT", "AGT") <= 4


def test_gc_skew_sign() -> None:
    s = gc_skew("GGGG" + "CCCC", 4)
    assert s[0] == 1.0 and s[-1] == -1.0


def test_orf_finds_planted() -> None:
    seq = "CCCATGAAAAAATAACCC"
    assert longest_orf(seq) == 12


def test_orf_none_when_no_atg() -> None:
    assert longest_orf("CCCCCCCCCC") == 0


def test_kmer_counts() -> None:
    spec = kmer_spectrum("AAAAA", 2)
    assert spec == {"AA": 4}


def test_column_ic_conserved() -> None:
    aln = ["AC", "AC", "AC"]
    assert column_ic(aln, 0) == 2.0


def test_column_ic_uniform() -> None:
    aln = ["A", "C", "G", "T"]
    assert abs(column_ic(aln, 0)) < 1e-9
