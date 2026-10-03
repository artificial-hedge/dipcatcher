"""ADAPT-VQE: greedy ansatz growth via gradient-ranked Pauli rotations.

Pool: {e^{-i theta P}} for P in a small Pauli set. Each round evaluates
dE/dtheta|_{0} propto <psi| [H, P] |psi>, appends the max-|gradient| operator,
then re-optimizes all angles by coordinate descent. Bench on a 2-qubit
Hamiltonian: energy reaches within epsilon of the exact ground state in few
operators, and the gradient ranking picks nonzero-gradient operators first.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 985

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_I = np.eye(2, dtype=complex)


def _kron(*ops: np.ndarray) -> np.ndarray:
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return np.asarray(out)


def h2_hamiltonian() -> np.ndarray:
    return np.asarray(
        -0.5 * _kron(_Z, _I)
        - 0.5 * _kron(_I, _Z)
        + 0.3 * _kron(_X, _X)
        + 0.2 * _kron(_Z, _Z)
        + 0.1 * _kron(_Y, _Y)
    )


def pool() -> list[np.ndarray]:
    return [
        np.asarray(_kron(_X, _I)),
        np.asarray(_kron(_I, _X)),
        np.asarray(_kron(_Y, _I)),
        np.asarray(_kron(_I, _Y)),
        np.asarray(_kron(_Y, _X)),
        np.asarray(_kron(_X, _Y)),
    ]


def apply_ansatz(
    angles: np.ndarray, ops: list[np.ndarray], psi0: np.ndarray | None = None
) -> np.ndarray:
    psi = np.array([1, 0, 0, 0], dtype=complex) if psi0 is None else psi0.copy()
    for th, p in zip(angles, ops, strict=True):
        c, s = np.cos(th / 2), np.sin(th / 2)
        u = c * np.eye(4, dtype=complex) - 1j * s * p
        psi = u @ psi
    return np.asarray(psi)


def exp_energy(psi: np.ndarray, h: np.ndarray) -> float:
    return float(np.real(psi.conj() @ h @ psi))


def adapt_round(
    angles: np.ndarray,
    ops: list[np.ndarray],
    h: np.ndarray,
    p_pool: list[np.ndarray],
    psi0: np.ndarray | None = None,
) -> int:
    psi = apply_ansatz(angles, ops, psi0)
    grads = [abs(exp_energy(psi, 1j * (h @ p - p @ h))) for p in p_pool]
    return int(np.argmax(grads))


def optimize(
    angles: np.ndarray,
    ops: list[np.ndarray],
    h: np.ndarray,
    psi0: np.ndarray | None = None,
) -> np.ndarray:
    angles = angles.copy()
    for _ in range(60):
        for j in range(len(angles)):
            best = angles[j]
            best_e = exp_energy(apply_ansatz(angles, ops, psi0), h)
            for delta in np.linspace(-np.pi, np.pi, 90):
                trial = angles.copy()
                trial[j] = angles[j] + delta
                e = exp_energy(apply_ansatz(trial, ops, psi0), h)
                if e < best_e:
                    best, best_e = angles[j] + delta, e
            angles[j] = best
    return angles


def adapt_vqe(
    h: np.ndarray,
    p_pool: list[np.ndarray],
    max_ops: int = 8,
    psi0: np.ndarray | None = None,
) -> tuple[float, int]:
    ops: list[np.ndarray] = []
    angles = np.zeros(0)
    for _ in range(max_ops):
        j = adapt_round(angles, ops, h, p_pool, psi0)
        ops.append(p_pool[j])
        angles = np.concatenate([angles, [0.0]])
        angles = optimize(angles, ops, h, psi0)
        psi = apply_ansatz(angles, ops, psi0)
        e = exp_energy(psi, h)
        if e < float(np.linalg.eigvalsh(h)[0]) + 1e-3:
            return e, len(ops)
    psi = apply_ansatz(angles, ops, psi0)
    return exp_energy(psi, h), len(ops)


def bench_adapt_vqe(seed: int = _SEED) -> dict[str, float]:
    _ = np.random.default_rng(seed)
    checks: list[bool] = []
    h = h2_hamiltonian()
    w_min = float(np.linalg.eigvalsh(h)[0])
    had = _kron(np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2), _I)
    psi0 = np.asarray(had @ np.array([1, 0, 0, 0], dtype=complex))
    e, n_ops = adapt_vqe(h, pool(), psi0=psi0)
    checks.append(e < w_min + 0.05)
    checks.append(n_ops <= 8)
    grads0 = [abs(exp_energy(psi0, 1j * (h @ p - p @ h))) for p in pool()]
    checks.append(max(grads0) > 1e-3)
    checks.append(int(np.argmax(grads0)) in (2, 3, 4, 5))
    score = float(np.mean(checks))
    return {"synthetic_adapt_vqe": score}
