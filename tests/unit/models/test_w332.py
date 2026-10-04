"""Wave-332 quantum-information module unit tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.bell_ineq import chsh_expect
from quant_fund.models.density_matrix import fidelity, partial_trace, pure_state, purity
from quant_fund.models.entanglement import concurrence, is_separable, negativity
from quant_fund.models.povm_measure import is_povm, projective
from quant_fund.models.qchannel import amplitude_damping, apply, is_tp
from quant_fund.models.state_tomo import bloch, from_bloch, reconstruct_1q


def test_density() -> None:
    bell = np.array([1, 0, 0, 1], complex) / np.sqrt(2)
    rho = pure_state(bell)
    assert abs(purity(rho) - 1.0) < 1e-9
    assert np.allclose(partial_trace(rho, [0], [2, 2]), np.eye(2) / 2)
    assert abs(fidelity(rho, rho) - 1.0) < 1e-9


def test_povm() -> None:
    z = [projective(np.array([1, 0], complex)), projective(np.array([0, 1], complex))]
    assert is_povm(z)
    assert not is_povm([z[0]])


def test_channel() -> None:
    assert is_tp(amplitude_damping(0.3))
    exc = np.array([[0, 0], [0, 1]], complex)
    assert abs(float(apply(exc, amplitude_damping(0.5))[1, 1].real) - 0.5) < 1e-9


def test_ent() -> None:
    bell = np.array([1, 0, 0, 1], complex) / np.sqrt(2)
    rb = np.outer(bell, bell.conj())
    assert abs(concurrence(rb) - 1.0) < 1e-9
    assert abs(negativity(rb) - 0.5) < 1e-9
    assert not is_separable(rb)


def test_chsh() -> None:
    import math

    bell = np.array([1, 0, 0, 1], complex) / np.sqrt(2)
    rho = np.outer(bell, bell.conj())
    s = chsh_expect(rho, 0.0, math.pi / 2, math.pi / 4, -math.pi / 4)
    assert abs(abs(s) - 2 * math.sqrt(2)) < 1e-9


def test_tomo() -> None:
    rho = np.array([[0.8, 0.3], [0.3, 0.2]], dtype=complex)
    assert np.allclose(from_bloch(bloch(rho)), rho, atol=1e-9)
    rec = reconstruct_1q({"x": (60, 40), "y": (50, 50), "z": (80, 20)})
    assert abs(float(rec[0, 0].real) - 0.8) < 1e-9
