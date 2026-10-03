"""Wave-316 quantum-3 module unit tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.adapt_vqe import adapt_round, apply_ansatz, exp_energy, h2_hamiltonian, pool
from quant_fund.models.hhl_lite import hhl_solve
from quant_fund.models.qdrift import qdrift_unitary, terms
from quant_fund.models.shadow_tomography import bell_state, measure_snapshot
from quant_fund.models.trotter_suzuki import err_vs_step, s2, s4, two_body_hamiltonian
from quant_fund.models.vqd_states import ansatz, coord_descent, energy


def test_product_formulas_unitarity() -> None:
    _, a, b = two_body_hamiltonian()
    for u in (s2(a, b, 0.3), s4(a, b, 0.3)):
        assert np.allclose(u.conj().T @ u, np.eye(4), atol=1e-10)


def test_err_decreases_with_order() -> None:
    _, a, b = two_body_hamiltonian()
    for r in (4, 8):
        assert err_vs_step("s4", a, b, 0.5, r) < err_vs_step("s2", a, b, 0.5, r)


def test_qdrift_is_unitary() -> None:
    rng = np.random.default_rng(0)
    hs, ps = terms()
    u = qdrift_unitary(hs, ps, 1.0, 50, rng)
    assert np.allclose(u.conj().T @ u, np.eye(4), atol=1e-10)


def test_shadow_snapshot_trace_one() -> None:
    rng = np.random.default_rng(1)
    s = measure_snapshot(bell_state(), 2, rng)
    assert abs(np.trace(s) - 1.0) < 1e-9


def test_vqd_ansatz_normalizes() -> None:
    psi = ansatz(np.array([0.3, -0.7, 1.1, 0.2]))
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-12
    h = h2_hamiltonian()
    th = coord_descent(np.zeros(4), h, None, 0.0)
    e = energy(th, h, None, 0.0)
    assert e <= float(np.linalg.eigvalsh(h)[0]) + 0.05


def test_adapt_round_picks_max_gradient() -> None:
    h = h2_hamiltonian()
    had = np.kron(np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2), np.eye(2))
    psi0 = np.asarray(had @ np.array([1, 0, 0, 0], dtype=complex))
    j = adapt_round(np.zeros(0), [], h, pool(), psi0)
    grads = [abs(exp_energy(psi0, 1j * (h @ p - p @ h))) for p in pool()]
    assert j == int(np.argmax(grads))


def test_apply_ansatz_prep_state() -> None:
    psi = apply_ansatz(np.zeros(2), [np.eye(4), np.eye(4)], np.array([0, 1, 0, 0], dtype=complex))
    assert np.allclose(psi, np.array([0, 1, 0, 0]))


def test_hhl_rejects_offgrid() -> None:
    rng = np.random.default_rng(0)
    q = rng.random((2, 2)) * 0.1
    bad = np.eye(2) + q + q.T  # eigenvalues off dyadic grid w.h.p.
    try:
        hhl_solve(bad, np.array([1.0, 0.0]), rng)
        raised = False
    except ValueError:
        raised = True
    assert raised
