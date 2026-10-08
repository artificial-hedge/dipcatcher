"""qDRIFT randomized Hamiltonian compilation (SYNTHETIC).

For H = sum_j h_j H_j (h_j > 0), qDRIFT samples L terms i.i.d. proportional to
h_j and applies prod_j exp(-i H_{s_j} tau), tau = t * lambda / L where
lambda = sum_j h_j. The expected channel equals the exact evolution to first
order with variance shrinking as t^2 lambda^2 / L. Bench: Monte-Carlo
expectation of a Pauli observable converges to the exact value with bias
vanishing vs a deterministic Trotter step's nonzero bias.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 982

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron(*ops: np.ndarray) -> np.ndarray:
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return np.asarray(out)


def terms() -> tuple[list[float], list[np.ndarray]]:
    ps = [
        _kron(_Z, _Z),
        _kron(_X, _I),
        _kron(_I, _X),
        _kron(_X, _X),
        _kron(_Y, _Y),
    ]
    hs = [1.0, 0.8, 0.8, 0.6, 0.4]
    return hs, [np.asarray(p) for p in ps]


def _uexp(m: np.ndarray, t: float) -> np.ndarray:
    w, v = np.linalg.eigh(m)
    return np.asarray(v @ np.diag(np.exp(-1j * t * w)) @ v.conj().T)


def qdrift_unitary(
    hs: list[float], ps: list[np.ndarray], t: float, l_steps: int, rng: np.random.Generator
) -> np.ndarray:
    lam = sum(hs)
    probs = np.array(hs) / lam
    u = np.eye(ps[0].shape[0], dtype=complex)
    for _ in range(l_steps):
        j = int(rng.choice(len(ps), p=probs))
        u = _uexp(ps[j], t * lam / l_steps) @ u
    return np.asarray(u)


def trotter_unitary(hs: list[float], ps: list[np.ndarray], t: float, r: int) -> np.ndarray:
    hmat = sum(h * p for h, p in zip(hs, ps, strict=True))
    u = np.eye(ps[0].shape[0], dtype=complex)
    for _ in range(r):
        for h, p in zip(hs, ps, strict=True):
            u = _uexp(p, h * t / r) @ u
    _ = hmat
    return np.asarray(u)


def expectation(psi0: np.ndarray, u: np.ndarray, obs: np.ndarray) -> float:
    psi = u @ psi0
    return float(np.real(psi.conj() @ obs @ psi))


def bench_qdrift(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    checks: list[bool] = []
    hs, ps = terms()
    hmat = np.asarray(sum(h * p for h, p in zip(hs, ps, strict=True)))
    t = 0.8
    u_exact = _uexp(hmat, t)
    psi0 = np.zeros(4, dtype=complex)
    psi0[0] = 1.0
    obs = _kron(_Z, _I)
    exact = expectation(psi0, u_exact, obs)
    l_steps = 24
    est = np.mean(
        [expectation(psi0, qdrift_unitary(hs, ps, t, l_steps, rng), obs) for _ in range(400)]
    )
    checks.append(bool(abs(est - exact) < 0.08))
    biases = []
    for _ in range(6):
        e = np.mean(
            [expectation(psi0, qdrift_unitary(hs, ps, t, l_steps, rng), obs) for _ in range(300)]
        )
        biases.append(e - exact)
    checks.append(bool(abs(float(np.mean(biases))) < 0.08))
    trot = expectation(psi0, trotter_unitary(hs, ps, t, 2), obs)
    checks.append(bool(abs(trot - exact) > 1e-4))
    s1 = np.std(
        [expectation(psi0, qdrift_unitary(hs, ps, t, l_steps, rng), obs) for _ in range(200)]
    )
    s2 = np.std(
        [expectation(psi0, qdrift_unitary(hs, ps, t, 4 * l_steps, rng), obs) for _ in range(200)]
    )
    checks.append(bool(s2 < s1))
    score = float(np.mean(checks))
    return {"synthetic_qdrift": score}
