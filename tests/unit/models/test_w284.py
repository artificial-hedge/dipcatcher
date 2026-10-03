"""Unit tests for wave-284 bioinformatics-3 canon modules."""

import numpy as np

from quant_fund.models.band_align import _nw_band
from quant_fund.models.codon_usage import cai
from quant_fund.models.fitch_pars import fitch
from quant_fund.models.jc69_lik import lik_closed
from quant_fund.models.nj_tree import nj
from quant_fund.models.seed_extend import seed_extend


def test_nj_quartet():
    d = np.array(
        [[0, 0.3, 0.45, 0.55], [0.3, 0, 0.55, 0.65], [0.45, 0.55, 0, 0.4], [0.55, 0.65, 0.4, 0]]
    )
    assert nj(d, ["a", "b", "c", "d"]) == {frozenset(["a", "b"]), frozenset(["c", "d"])}


def test_fitch_zero():
    edges = [("u", "a"), ("u", "b"), ("v", "c"), ("v", "d"), ("r", "u"), ("r", "v")]
    assert fitch({"a": 0, "b": 0, "c": 0, "d": 0}, edges) == 0


def test_fitch_one():
    edges = [("u", "a"), ("u", "b"), ("v", "c"), ("v", "d"), ("r", "u"), ("r", "v")]
    assert fitch({"a": 0, "b": 0, "c": 0, "d": 1}, edges) == 1


def test_seed_extend_perfect():
    s, qs, ss = seed_extend("AAAACCCCGGGG", "TTCCCCGGTT", k=3)
    assert s >= 6 and qs <= 4


def test_band_vs_full():
    assert _nw_band("ACGTACGT", "ACGTTCGT", 3) == -1 + 7


def test_jc69_limits():
    assert abs(lik_closed("A", "A", 0.0, 0.0) - 0.25) < 1e-9
    assert abs(lik_closed("A", "C", 0.0, 0.0)) < 1e-12
    assert abs(lik_closed("A", "C", 50.0, 50.0) - 1 / 16) < 1e-6


def test_cai_max():
    w = {"AAA": 1.0, "AAC": 0.5}
    assert cai(["AAA", "AAA"], w) == 1.0
