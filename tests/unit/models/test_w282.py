"""Wave-282 combinatorics module tests."""

from quant_fund.models.gray_code import gray
from quant_fund.models.inversion_count import inversions
from quant_fund.models.latin_square import is_latin
from quant_fund.models.stirling_count import bell, stirling2
from quant_fund.models.subset_sum_dp import reachable


def test_gray_len() -> None:
    assert len(gray(5)) == 32


def test_stirling_edge() -> None:
    assert stirling2(5, 1) == 1 and stirling2(5, 5) == 1 and bell(4) == 15


def test_inversions_sorted() -> None:
    assert inversions([1, 2, 3]) == 0


def test_subset_trivial() -> None:
    assert reachable([3, 5], 8) and not reachable([4], 3)


def test_latin_detects_dup() -> None:
    bad = [[1, 1], [2, 2]]
    assert not is_latin(bad, 2)
