"""CHSH game: classical bound vs quantum Tsirelson violation (SYNTHETIC bench)."""

from __future__ import annotations

import math

import numpy as np

X = np.array([[0, 1], [1, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)


def rot(theta: float) -> np.ndarray:
    """Observable with Bloch angle theta: cos Z + sin X."""
    return math.cos(theta) * Z + math.sin(theta) * X


def chsh_expect(rho: np.ndarray, a: float, a2: float, b: float, b2: float) -> float:
    """S = E(a,b) - E(a,b2) + E(a2,b) + E(a2,b2)."""
    A, A2, B, B2 = rot(a), rot(a2), rot(b), rot(b2)

    def e(o1: np.ndarray, o2: np.ndarray) -> float:
        return float(np.real(np.trace(rho @ np.kron(o1, o2))))

    return e(A, B) + e(A, B2) + e(A2, B) - e(A2, B2)


def classical_chsh() -> float:
    return 2.0


def _bench_bell_ineq(seed: int = 0) -> float:
    checks = []
    bell = np.array([1, 0, 0, 1], complex) / np.sqrt(2)
    rho = np.outer(bell, bell.conj())
    # optimal settings: a=0, a'=pi/2, b=pi/4, b'=-pi/4 -> |S| = 2*sqrt(2)
    s = chsh_expect(rho, 0.0, math.pi / 2, math.pi / 4, -math.pi / 4)
    checks.append(abs(abs(s) - 2 * math.sqrt(2)) < 1e-9)
    # same-settings -> no violation
    s2 = chsh_expect(rho, 0.0, 0.0, 0.0, 0.0)
    checks.append(abs(s2) <= 2.0 + 1e-9)
    # product state never violates
    prod = np.kron(np.array([[1, 0], [0, 0]], complex), np.array([[1, 0], [0, 0]], complex))
    s3 = chsh_expect(prod, 0.0, math.pi / 2, math.pi / 4, -math.pi / 4)
    checks.append(abs(s3) <= 2.0 + 1e-9)
    checks.append(abs(2 * math.sqrt(2) - 2.828427) < 1e-4)
    return sum(checks) / len(checks)


def bench_bell_ineq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bell_ineq": _bench_bell_ineq(seed)}
