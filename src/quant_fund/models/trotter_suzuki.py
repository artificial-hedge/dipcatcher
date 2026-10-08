"""Suzuki–Trotter product formulas of orders 1, 2, 4 for Hamiltonian simulation (SYNTHETIC).

For H = A + B, U(t) = exp(-iHt) is approximated by alternating exponentials:
  S1(t) = e^{-iAt} e^{-iBt}
  S2(t) = e^{-iAt/2} e^{-iBt} e^{-iAt/2}            (Strang)
  S4(t) = S2(p t) S2(p t) S2((1-4p) t) S2(p t) S2(p t),  p = 1/(4-4^{1/3})
Step-error analysis on a 2-qubit Hamiltonian: measured slope of ||U_approx - U_exact||
vs step h recovers orders ~2, ~3, ~5 in operator norm.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 981

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron(*ops: np.ndarray) -> np.ndarray:
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return np.asarray(out)


def two_body_hamiltonian() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = _kron(_Z, _Z) + 0.3 * _kron(_Z, _I)
    b = _kron(_X, _I) + 0.7 * _kron(_X, _X)
    return np.asarray(a + b), np.asarray(a), np.asarray(b)


def _expm(m: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(m)
    return np.asarray(v @ np.diag(np.exp(w)) @ v.conj().T)


def _uexp(m: np.ndarray, t: float) -> np.ndarray:
    w, v = np.linalg.eigh(m)
    return np.asarray(v @ np.diag(np.exp(-1j * t * w)) @ v.conj().T)


def s1(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return np.asarray(_uexp(a, t) @ _uexp(b, t))


def s2(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return np.asarray(_uexp(a, t / 2) @ _uexp(b, t) @ _uexp(a, t / 2))


def s4(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
    return np.asarray(
        s2(a, b, p * t)
        @ s2(a, b, p * t)
        @ s2(a, b, (1.0 - 4.0 * p) * t)
        @ s2(a, b, p * t)
        @ s2(a, b, p * t)
    )


def err_vs_step(formula: str, a: np.ndarray, b: np.ndarray, t: float, r: int) -> float:
    h = t / r
    fn = {"s1": s1, "s2": s2, "s4": s4}[formula]
    u = fn(a, b, h)
    for _ in range(r - 1):
        u = u @ fn(a, b, h)
    exact = _uexp(a + b, t)
    return float(np.linalg.norm(u - exact, 2))


def bench_trotter_suzuki(seed: int = _SEED) -> dict[str, float]:
    _ = np.random.default_rng(seed)
    checks: list[bool] = []
    _, a, b = two_body_hamiltonian()
    t = 0.5
    for name, lo, hi in [("s1", 0.8, 1.4), ("s2", 1.8, 2.4), ("s4", 3.5, 4.6)]:
        e1 = err_vs_step(name, a, b, t, 4)
        e2 = err_vs_step(name, a, b, t, 8)
        e3 = err_vs_step(name, a, b, t, 16)
        slope = float(np.log(e1 / e3) / np.log(4.0))
        checks.append(lo < slope < hi)
        checks.append(e1 > e2 > e3)
    checks.append(
        err_vs_step("s4", a, b, t, 8)
        < err_vs_step("s2", a, b, t, 8)
        < err_vs_step("s1", a, b, t, 8)
    )
    score = float(np.mean(checks))
    return {"synthetic_trotter_suzuki": score}
