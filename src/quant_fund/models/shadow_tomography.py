"""Classical shadows (Huang–Kueng–Preskill): random-Pauli measurement snapshots (SYNTHETIC).

Each qubit is measured in a uniformly random Pauli basis; the snapshot is
rho_hat = tensor_j (3 |b_j><b_j| - I)  with E[rho_hat] = rho. Expectation of a
Pauli observable is estimated by median-of-means over K snapshots. Bench on a
Bell state: Z0, Z1 and X0 X1 estimates converge to truth as K grows, and the
estimator is unbiased across independent runs.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 983

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS = [_X, _Y, _Z]


def _kron(*ops: np.ndarray) -> np.ndarray:
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return np.asarray(out)


def _eigvec(pauli: np.ndarray, outcome: int) -> np.ndarray:
    w, v = np.linalg.eigh(pauli)
    return np.asarray(v[:, 1 if outcome > 0 else 0])


def measure_snapshot(rho: np.ndarray, n_qubits: int, rng: np.random.Generator) -> np.ndarray:
    bases = [int(rng.integers(3)) for _ in range(n_qubits)]
    eig_lists = []
    for q in range(n_qubits):
        _, v = np.linalg.eigh(_PAULIS[bases[q]])
        eig_lists.append([v[:, 0], v[:, 1]])
    probs = []
    states = []
    for idx in range(2**n_qubits):
        bits = [(idx >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        vec = _kron(*[eig_lists[q][bits[q]] for q in range(n_qubits)])
        states.append(vec)
        probs.append(float(np.real(vec.conj() @ rho @ vec)))
    probs_arr = np.clip(np.asarray(probs, dtype=np.float64), 0, 1)
    probs_arr = probs_arr / probs_arr.sum()
    idx = int(rng.choice(2**n_qubits, p=probs_arr))
    bits = [(idx >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
    factors = []
    for q in range(n_qubits):
        b = eig_lists[q][bits[q]]
        factors.append(3 * np.outer(b, b.conj()) - np.eye(2))
    return np.asarray(_kron(*factors))


def _projector(pauli: np.ndarray, q: int, outcome: int, n: int) -> np.ndarray:
    b = _eigvec(pauli, outcome)
    proj = np.outer(b, b.conj())
    ops = [np.eye(2) for _ in range(n)]
    ops[q] = proj
    return np.asarray(_kron(*ops))


def shadow_estimate(snapshots: list[np.ndarray], obs: np.ndarray, k_groups: int = 5) -> float:
    vals = np.array([np.real(np.trace(s @ obs)) for s in snapshots])
    groups = np.array_split(vals, k_groups)
    return float(np.median([g.mean() for g in groups]))


def bell_state() -> np.ndarray:
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    return np.asarray(np.outer(psi, psi.conj()))


def bench_shadow_tomography(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    checks: list[bool] = []
    rho = bell_state()
    z0 = _kron(_Z, np.eye(2))
    xx = _kron(_X, _X)
    for k, tol in [(50, 0.6), (400, 0.25)]:
        snaps = [measure_snapshot(rho, 2, rng) for _ in range(k)]
        e_z = shadow_estimate(snaps, z0)
        e_x = shadow_estimate(snaps, xx)
        checks.append(abs(e_z - 0.0) < tol and abs(e_x - 1.0) < tol)
    runs = []
    for _ in range(8):
        snaps = [measure_snapshot(rho, 2, rng) for _ in range(200)]
        runs.append(shadow_estimate(snaps, xx))
    checks.append(abs(float(np.mean(runs)) - 1.0) < 0.2)
    zz = _kron(_Z, _Z)
    runs_z = []
    for _ in range(8):
        snaps = [measure_snapshot(rho, 2, rng) for _ in range(200)]
        runs_z.append(shadow_estimate(snaps, zz))
    checks.append(abs(float(np.mean(runs_z)) - 1.0) < 0.2)
    score = float(np.mean(checks))
    return {"synthetic_shadow_tomography": score}
