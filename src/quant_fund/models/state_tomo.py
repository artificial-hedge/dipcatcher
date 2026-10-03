"""Quantum state tomography from Pauli-basis measurement stats (SYNTHETIC bench)."""

from __future__ import annotations

import numpy as np

X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)
PAULI = [np.eye(2), X, Y, Z]


def bloch(rho: np.ndarray) -> np.ndarray:
    out: np.ndarray = np.array(
        [float(np.real(np.trace(rho @ p))) for p in (X, Y, Z)], dtype=float
    )
    return out


def from_bloch(r: np.ndarray) -> np.ndarray:
    out: np.ndarray = np.asarray(
        0.5 * (np.eye(2) + r[0] * X + r[1] * Y + r[2] * Z), dtype=complex
    )
    return out


def reconstruct_1q(counts: dict[str, tuple[int, int]]) -> np.ndarray:
    """counts: {"x":(n_plus,n_minus), "y":..., "z":...} -> rho."""
    r = np.zeros(3)
    for i, k in enumerate(("x", "y", "z")):
        np_, nm = counts[k]
        r[i] = (np_ - nm) / max(1, np_ + nm)
    return from_bloch(r)


def reconstruct_2q(expvals: dict[tuple[int, int], float]) -> np.ndarray:
    """rho = 1/4 sum_{i,j} <sigma_i x sigma_j> sigma_i x sigma_j."""
    rho = np.zeros((4, 4), complex)
    for i in range(4):
        for j in range(4):
            rho = rho + expvals.get((i, j), 0.0) * np.kron(PAULI[i], PAULI[j])
    return rho / 4


def expect_2q(rho: np.ndarray, i: int, j: int) -> float:
    return float(np.real(np.trace(rho @ np.kron(PAULI[i], PAULI[j]))))


def _bench_state_tomo(seed: int = 0) -> float:
    checks = []
    rho = np.array([[0.8, 0.3], [0.3, 0.2]], dtype=complex)
    r = bloch(rho)
    checks.append(np.allclose(from_bloch(r), rho, atol=1e-9))
    counts = {"x": (60, 40), "y": (50, 50), "z": (80, 20)}
    rec = reconstruct_1q(counts)
    checks.append(abs(float(rec[0, 0].real) - 0.8) < 1e-9)
    bell = np.array([1, 0, 0, 1], complex) / np.sqrt(2)
    rb = np.outer(bell, bell.conj())
    ev = {(i, j): expect_2q(rb, i, j) for i in range(4) for j in range(4)}
    rec2 = reconstruct_2q(ev)
    checks.append(np.allclose(rec2, rb, atol=1e-9))
    checks.append(abs(expect_2q(rb, 1, 1) - 1.0) < 1e-9)  # <XX>=1 on Phi+
    checks.append(abs(expect_2q(rb, 3, 3) - 1.0) < 1e-9)  # <ZZ>=1
    return sum(checks) / len(checks)


def bench_state_tomo(seed: int = 0) -> dict[str, float]:
    return {"synthetic_state_tomo": _bench_state_tomo(seed)}
