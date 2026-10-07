import numpy as np
import pytest

from quant_fund.flowbars.sample_weights import (
    avg_uniqueness,
    label_overlap_matrix,
    sequential_bootstrap,
    uniqueness_weights,
)

pytestmark = pytest.mark.synthetic


def test_overlap_matrix_hand_example() -> None:
    t0 = np.array([0, 2, 4, 6])
    t1 = np.array([3, 5, 7, 8])
    m = label_overlap_matrix(t0, t1)
    # spans: [0,3),[2,5),[4,7),[6,8)
    expected = np.array(
        [
            [True, True, False, False],
            [True, True, True, False],
            [False, True, True, True],
            [False, False, True, True],
        ]
    )
    np.testing.assert_array_equal(m, expected)


def test_touching_ends_do_not_overlap() -> None:
    t0 = np.array([0, 3])
    t1 = np.array([3, 6])
    m = label_overlap_matrix(t0, t1)
    assert not m[0, 1]
    assert m[0, 0] and m[1, 1]


def test_avg_uniqueness_disjoint_is_one() -> None:
    t0 = np.array([0, 2, 4])
    t1 = np.array([1, 3, 5])
    u = avg_uniqueness(t0, t1)
    np.testing.assert_allclose(u, np.ones(3))


def test_avg_uniqueness_overlapped_is_lower() -> None:
    t0 = np.array([0, 1])
    t1 = np.array([4, 4])
    u = avg_uniqueness(t0, t1)
    # label0 lives at t=0..3 with concurrency 1,2,2,2 → mean(1, 1/2, 1/2, 1/2)
    # label1 lives at t=1..3 with concurrency 2,2,2
    np.testing.assert_allclose(u, [0.625, 0.5])


def test_sequential_bootstrap_deterministic() -> None:
    t0 = np.array([0, 1, 2, 3, 4, 5])
    t1 = np.array([3, 3, 6, 6, 8, 8])
    m = label_overlap_matrix(t0, t1)
    a = sequential_bootstrap(m, 200, seed=42)
    b = sequential_bootstrap(m, 200, seed=42)
    np.testing.assert_array_equal(a, b)
    assert set(a).issubset(set(range(6)))


def test_sequential_bootstrap_prefers_low_overlap() -> None:
    # labels 0-3 are mutually disjoint; label 4 spans all of them. The
    # sequential bootstrap should draw the disjoint labels more often.
    t0 = np.array([0, 2, 4, 6, 0])
    t1 = np.array([1, 3, 5, 7, 7])
    m = label_overlap_matrix(t0, t1)
    draws = sequential_bootstrap(m, 4000, seed=7)
    counts = np.bincount(draws, minlength=5).astype(np.float64)
    freq = counts / counts.sum()
    assert np.mean(freq[:4]) > freq[4]


def test_uniqueness_weights_sum_to_mean_one() -> None:
    u = np.array([1.0, 0.5, 0.25, 0.75])
    w = uniqueness_weights(u)
    assert w.sum() == pytest.approx(len(u))
    assert w[0] > w[2]


def test_uniqueness_weights_with_attribution() -> None:
    u = np.array([1.0, 1.0])
    y = np.array([0.1, 2.0])
    w = uniqueness_weights(u, y)
    assert w[1] > w[0]


def test_uniqueness_weights_all_zero_uniform() -> None:
    w = uniqueness_weights(np.zeros(4))
    np.testing.assert_allclose(w, np.ones(4))
