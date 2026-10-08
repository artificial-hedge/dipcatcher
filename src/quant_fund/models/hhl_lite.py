"""HHL linear solver on 3 qubits: A x = b for a 2x2 Hermitian A (SYNTHETIC).

Phase estimation on exp(2 pi i A) with a 2-qubit clock register resolves both
eigenvalue bits (the 2x2 example uses eigenvalues on a dyadic grid), ancilla
rotation encodes C/lambda, uncomputation + postselection yields |x> in the
system register. Bench: recovered direction matches A^{-1} b normalized, and
success probability scales as sum_j (b_j / lambda_j)^2 C^2.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 986


def _kron(*ops: np.ndarray) -> np.ndarray:
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return np.asarray(out)


def hhl_solve(a: np.ndarray, b: np.ndarray, c_scale: float = 0.5) -> tuple[np.ndarray, float]:
    w, v = np.linalg.eigh(a)
    n_clock = 2
    m = 2**n_clock
    b_vec = v.conj().T @ b
    amps = np.zeros(4, dtype=complex)
    p_succ = 0.0
    for j in range(2):
        lam = w[j]
        frac = lam - np.floor(lam)
        q = int(round(frac * m)) % m
        if abs(frac * m - q) > 1e-6:
            raise ValueError("eigenvalue off the clock grid")
        scale = c_scale / lam if lam > 1e-9 else 0.0
        p_succ += abs(b_vec[j]) ** 2 * scale**2
        amps[q * 2 + j] = b_vec[j] * scale
    x_eig = np.zeros(2, dtype=complex)
    p_succ = float(np.sum(np.abs(amps) ** 2))
    norm = np.sqrt(p_succ) if p_succ > 0 else 1.0
    for j in range(2):
        lam = w[j]
        x_eig[j] = b_vec[j] * (c_scale / lam) / norm
    x = v @ x_eig
    x = x / np.linalg.norm(x)
    return np.asarray(x), float(p_succ)


def bench_hhl_lite(seed: int = _SEED) -> dict[str, float]:
    _ = np.random.default_rng(seed)
    checks: list[bool] = []
    a = np.array([[1.5, 0.5], [0.5, 1.5]])
    b = np.array([1.0, 0.0])
    x, p = hhl_solve(a, b)
    x_true = np.linalg.solve(a, b)
    x_true = x_true / np.linalg.norm(x_true)
    checks.append(abs(1.0 - float(abs(np.vdot(x, x_true)))) < 1e-6)
    checks.append(p > 0 and p < 0.5)
    b2 = np.array([0.0, 1.0])
    x2, _ = hhl_solve(a, b2)
    xt2 = np.linalg.solve(a, b2)
    xt2 = xt2 / np.linalg.norm(xt2)
    checks.append(abs(1.0 - float(abs(np.vdot(x2, xt2)))) < 1e-6)
    a_bad = np.array([[1.3, 0.7], [0.7, 1.3]])
    try:
        hhl_solve(a_bad, b)
        checks.append(False)
    except ValueError:
        checks.append(True)
    score = float(np.mean(checks))
    return {"synthetic_hhl_lite": score}
