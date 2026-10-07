"""Adversarial probes for _ttc_synth (SYNTHETIC)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._ttc_synth import (
    eval_chain,
    op_features,
    synth_problems,
    true_ops,
)


def test_true_ops_learnable_map():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    f = true_ops(a)
    np.testing.assert_array_equal(f[0], [(1 + 2 + 0) % 3, (2 + 3 + 1) % 3])
    np.testing.assert_array_equal(f[1], [(4 + 5 + 0) % 3, (5 + 6 + 1) % 3])


@pytest.mark.parametrize("a", [np.zeros((4, 2), dtype=np.int64), np.zeros(3, dtype=np.int64)])
def test_true_ops_hostile(a):
    with pytest.raises(ValueError):
        true_ops(a)


def test_eval_chain_semantics():
    a = np.array([[2, 3, 4]])
    ops_add = np.array([[0, 0]])
    ops_mul = np.array([[2, 2]])
    assert eval_chain(a, ops_add)[0] == pytest.approx(2 + 3 + 4)
    assert eval_chain(a, ops_mul)[0] == pytest.approx(2 * 3 * 4)
    ops_sub = np.array([[1, 1]])
    assert eval_chain(a, ops_sub)[0] == pytest.approx(2 - 3 - 4)


def test_eval_chain_invalid_op_laundered_to_multiply():
    a = np.array([[2, 3, 4]])
    bad = np.array([[0, 7]])  # op 7 must NOT silently become multiply
    with pytest.raises(ValueError):
        eval_chain(a, bad)
    with pytest.raises(ValueError):
        eval_chain(a, np.array([[-1, 0]]))


def test_eval_chain_shape_mismatch():
    a = np.array([[2, 3, 4]])
    with pytest.raises(ValueError):
        eval_chain(a, np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(ValueError):
        eval_chain(a, np.zeros((1, 3), dtype=np.int64))


def test_synth_problems_deterministic_and_consistent():
    rng = np.random.default_rng(0)
    a1, o1, v1 = synth_problems(32, rng)
    rng = np.random.default_rng(0)
    a2, o2, v2 = synth_problems(32, rng)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(v1, v2)
    # v == eval_chain(a, true_ops(a)) by construction
    np.testing.assert_array_equal(v1, eval_chain(a1, o1))


def test_synth_problems_hostile():
    with pytest.raises(ValueError):
        synth_problems(0, np.random.default_rng(0))


def test_op_features_recovers_op_linearly():
    rng = np.random.default_rng(0)
    a, ops, _ = synth_problems(300, rng)
    for step in (0, 1):
        x = op_features(a, step)
        assert x.shape == (300, 10)
        # sin/cos features of (a_s + a_{s+1} + step) mod 3 → linear probe separable
        w = np.linalg.lstsq(np.hstack([x, np.ones((300, 1))]), np.eye(3)[ops[:, step]], rcond=None)[
            0
        ]
        pred = np.argmax(np.hstack([x, np.ones((300, 1))]) @ w, 1)
        assert (pred == ops[:, step]).mean() > 0.95


@pytest.mark.parametrize("step", [-1, 2, 5])
def test_op_features_hostile_step(step):
    a = np.ones((4, 3), dtype=np.int64)
    with pytest.raises(ValueError):
        op_features(a, step)


def test_op_features_hostile_shape():
    with pytest.raises(ValueError):
        op_features(np.ones((4, 4), dtype=np.int64), 0)
