"""Wave-275 databases-4 module tests."""

import numpy as np

from quant_fund.models.columnar_scan import columnar_filter, project_sum
from quant_fund.models.graceful_hash import graceful_join
from quant_fund.models.index_intersect import intersect_sorted
from quant_fund.models.late_materialize import late_materialize
from quant_fund.models.radix_join import radix_join
from quant_fund.models.simd_filter import simd_count


def test_columnar_filter() -> None:
    cols = {"a": np.array([0.5, 1.5, 2.0]), "b": np.array([1.0, 2.0, 3.0])}
    mask = columnar_filter(cols, "a", 0.0, 1.0)
    assert mask.tolist() == [True, False, False]
    assert project_sum(cols, mask, ["b"])["b"] == 1.0


def test_simd_count_edges() -> None:
    arr = np.array([1.0, 2.0, 3.0])
    assert simd_count(arr, 1.5, 8) == 2
    assert simd_count(arr, 5.0, 8) == 0


def test_late_materialize() -> None:
    cols = {"a": np.array([1.0, -1.0, 2.0]), "b": np.array([0.5, 1.0, -1.0])}
    assert late_materialize(cols, [("a", 0.0), ("b", 0.0)]) == [0]


def test_radix_join_basic() -> None:
    got = {t[0] for t in radix_join(np.array([1, 2, 3]), np.array([2, 3, 4]))}
    assert got == {2, 3}


def test_radix_join_empty() -> None:
    assert radix_join(np.array([1]), np.array([2])) == []


def test_graceful_join() -> None:
    got = {t[0] for t in graceful_join(np.array([1, 2, 2, 3]), np.array([2, 3]), 1)}
    assert got == {2, 3}


def test_intersect_sorted() -> None:
    assert intersect_sorted([[1, 2, 3], [2, 3], [3, 4]]) == [3]


def test_intersect_empty_input() -> None:
    assert intersect_sorted([]) == []
