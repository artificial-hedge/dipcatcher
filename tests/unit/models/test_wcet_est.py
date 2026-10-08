import numpy as np

from quant_fund.models import wcet_est as we


def test_longest_path_crafted_dags() -> None:
    # chain: single path 0->1->2->3 sums everything
    chain = {0: [1], 1: [2], 2: [3], 3: []}
    assert we._longest_path(chain, np.array([2, 5, 1, 7])) == 15
    # fork picks the heavier branch, not both
    fork = {0: [1, 2], 1: [3], 2: [3], 3: []}
    assert we._longest_path(fork, np.array([1, 9, 3, 4])) == 14
    # unreachable nodes never counted
    leaf_only = {0: [], 1: [2], 2: []}
    assert we._longest_path(leaf_only, np.array([4, 9, 9])) == 4


def test_bench() -> None:
    assert we.bench_wcet_est()["synthetic_wcet_valid"] == 1.0
