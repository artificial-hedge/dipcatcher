"""Probes for _qc_synth (statevector fixture)."""

import numpy as np
import pytest

from quant_fund.models._qc_synth import (
    H,
    X,
    Z,
    apply1,
    init_state,
    maxcut_cost,
    maxcut_graph,
    measure_probs,
)


def test_init_state_is_computational_zero():
    psi = init_state(3)
    assert psi.shape == (8,)
    assert psi[0] == 1.0 and (np.abs(psi[1:]) == 0).all()


def test_init_state_rejects_zero_qubits():
    with pytest.raises(ValueError):
        init_state(0)


def test_apply1_x_flips_qubit():
    psi = apply1(init_state(2), X, 0, 2)  # qubit 0 = MSB? verify actual convention
    # whichever convention, exactly one basis state has prob 1
    probs = measure_probs(psi)
    assert probs.sum() == pytest.approx(1.0)
    assert probs.max() == pytest.approx(1.0)
    assert probs[0] == 0.0  # flipped away from |00...>


def test_apply1_hadmard_superposition():
    psi = apply1(init_state(1), H, 0, 1)
    probs = measure_probs(psi)
    np.testing.assert_allclose(probs, [0.5, 0.5])


@pytest.mark.parametrize(
    "gate,qubit,n,psi",
    [
        (X, -1, 2, init_state(2)),  # negative qubit silently wrapped before
        (X, 2, 2, init_state(2)),  # qubit out of range
        (X, 0, 0, np.ones(1)),  # n=0
        (np.eye(3), 0, 2, init_state(2)),  # 3x3 gate not a qubit gate
        (Z, 0, 3, init_state(2)),  # psi size mismatched to n
    ],
)
def test_apply1_hostile(gate, qubit, n, psi):
    with pytest.raises(ValueError):
        apply1(psi, gate, qubit, n)


def test_maxcut_graph_valid_edges_deterministic():
    e1 = maxcut_graph(0, n=4)
    e2 = maxcut_graph(0, n=4)
    assert e1 == e2
    assert all(0 <= i < j < 4 for i, j in e1)


def test_maxcut_graph_rejects_tiny_n():
    # n=1 previously fabricated edge (0,1) referencing a nonexistent node
    for n in (0, 1):
        with pytest.raises(ValueError):
            maxcut_graph(0, n=n)


def test_maxcut_cost_counts_cut():
    edges = [(0, 1), (1, 2), (2, 3)]
    assert maxcut_cost(0b0101, edges) == 3.0  # alternating bits cut all
    assert maxcut_cost(0b0000, edges) == 0.0
    with pytest.raises(ValueError):
        maxcut_cost(-1, edges)
