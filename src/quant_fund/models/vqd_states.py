"""Variational quantum deflation: ground + first-excited state via overlap penalty (SYNTHETIC).

Circuit ansatz on 2 qubits: RY on each, CNOT, RY on each (4 angles).
Ground state: coordinate-descent on <H>. Excited state: same descent on
<H> + beta |<psi|psi_0>|^2. Bench: E0 ~ lambda_min, E1 ~ next eigenvalue,
and the two states are near-orthogonal.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 984

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron(*ops: np.ndarray) -> np.ndarray:
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return np.asarray(out)


def heisenberg_h() -> np.ndarray:
    return np.asarray(_kron(_X, _X) + _kron(_Z, _Z) + 0.5 * _kron(_Z, np.eye(2)))


def _ry(t: float) -> np.ndarray:
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


_CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def ansatz(theta: np.ndarray) -> np.ndarray:
    u1 = _kron(_ry(theta[0]), _ry(theta[1]))
    u2 = _kron(_ry(theta[2]), _ry(theta[3]))
    psi = u2 @ _CNOT @ u1 @ np.array([1, 0, 0, 0], dtype=complex)
    return np.asarray(psi)


def energy(theta: np.ndarray, h: np.ndarray, ref: np.ndarray | None, beta: float) -> float:
    psi = ansatz(theta)
    e = float(np.real(psi.conj() @ h @ psi))
    if ref is not None:
        e += beta * float(abs(psi.conj() @ ref)) ** 2
    return e


def coord_descent(
    theta: np.ndarray, h: np.ndarray, ref: np.ndarray | None, beta: float, iters: int = 60
) -> np.ndarray:
    theta = theta.copy()
    for _ in range(iters):
        for j in range(4):
            for delta in np.linspace(-np.pi, np.pi, 60):
                trial = theta.copy()
                trial[j] += delta
                if energy(trial, h, ref, beta) < energy(theta, h, ref, beta):
                    theta = trial
    return theta


def bench_vqd_states(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    checks: list[bool] = []
    h = heisenberg_h()
    w = np.linalg.eigvalsh(h)
    th0 = coord_descent(rng.uniform(-np.pi, np.pi, 4), h, None, 0.0)
    e0 = energy(th0, h, None, 0.0)
    checks.append(abs(e0 - float(w[0])) < 0.02)
    psi0 = ansatz(th0)
    th1 = coord_descent(rng.uniform(-np.pi, np.pi, 4), h, psi0, beta=4.0)
    psi1 = ansatz(th1)
    e1 = float(np.real(psi1.conj() @ h @ psi1))
    checks.append(abs(e1 - float(w[1])) < 0.1)
    checks.append(float(abs(psi1.conj() @ psi0)) ** 2 < 0.05)
    th0b = coord_descent(np.zeros(4), h, None, 0.0)
    checks.append(energy(th0b, h, None, 0.0) < energy(np.zeros(4), h, None, 0.0))
    score = float(np.mean(checks))
    return {"synthetic_vqd_states": score}
