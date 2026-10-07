import numpy as np
import pytest

from quant_fund.models.belief_propagation import (
    bench_belief_propagation,
    bp_chain,
    bp_loopy,
    brute_marginals,
)


def test_chain_exact():
    rng = np.random.default_rng(0)
    n, k = 5, 2
    pn = np.exp(0.3 * rng.standard_normal((n, k)))
    pn /= pn.sum(1, keepdims=True)
    pe = np.exp(0.4 * rng.standard_normal((n - 1, k, k)))
    bp = bp_chain(pn, pe)
    ex = brute_marginals(pn, [(i, i + 1) for i in range(n - 1)], pe)
    assert np.abs(bp - ex).max() < 1e-6


def test_loopy_runs_simplex():
    rng = np.random.default_rng(1)
    n, k = 4, 2
    pn = np.exp(0.2 * rng.standard_normal((n, k)))
    pn /= pn.sum(1, keepdims=True)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    pe = np.exp(0.3 * rng.standard_normal((4, k, k)))
    bp = bp_loopy(pn, edges, pe)
    assert np.allclose(bp.sum(1), 1.0)


def test_loopy_edgeless_returns_node_marginals():
    # an edge-free MRF: marginals are the normalized node potentials;
    # must not crash on the empty message set (old code did max() of empty).
    pn = np.array([[0.7, 0.3], [0.2, 0.8]])
    out = bp_loopy(pn, [], np.zeros((0, 2, 2)))
    np.testing.assert_allclose(out, pn)


def test_brute_marginals_rejects_zero_mass():
    pn = np.zeros((2, 2))
    pe = np.zeros((1, 2, 2))
    with pytest.raises(ValueError, match="nonpositive total mass"):
        brute_marginals(pn, [(0, 1)], pe)


def test_bench_keys():
    out = bench_belief_propagation()
    assert out["synthetic_bp_chain_l1"] < 1e-6
    assert out["synthetic_bp_loopy_l1"] < 0.1
    assert out["synthetic_bp_chain_simplex_err"] < 1e-9
