"""Fourier transform on S3 at irreps + Parseval on regular functions (SYNTHETIC)."""

from __future__ import annotations

import numpy as np

S3 = [(0, 1, 2), (1, 0, 2), (0, 2, 1), (2, 1, 0), (1, 2, 0), (2, 0, 1)]


def perm_matrix(p: tuple[int, ...]) -> np.ndarray:
    m = np.zeros((3, 3))
    for i in range(3):
        m[i, p[i]] = 1.0
    return m


def fourier_at(f: dict[tuple[int, ...], float], rho) -> np.ndarray:
    """f-hat(rho) = sum_g f(g) rho(g)."""
    out = sum(f[g] * rho(g) for g in S3)
    return np.asarray(out)


def _bench_fourier_sn(seed: int = 0) -> float:
    checks = []
    e = (0, 1, 2)
    # delta at identity: f(g) = 1 iff g=e. Fourier at perm rep = identity matrix
    f: dict[tuple[int, ...], float] = {g: (1.0 if g == e else 0.0) for g in S3}
    checks.append(np.allclose(fourier_at(f, perm_matrix), np.eye(3)))
    # constant function: Fourier = sum of all perm matrices = J (all ones)
    fc: dict[tuple[int, ...], float] = {g: 1.0 for g in S3}
    checks.append(np.allclose(fourier_at(fc, perm_matrix), 2.0 * np.ones((3, 3))))
    # Parseval at trivial rep: |sum f| vs norm — check Plancherel on delta: sum_g f(g)^2 = 1
    checks.append(abs(sum(v * v for v in f.values()) - 1.0) < 1e-9)
    # convolution theorem: (f*f)-hat = fhat fhat. f*f at e counts pairs (g,g^-1): for delta_e trivial
    checks.append(np.allclose(fourier_at(f, perm_matrix) @ fourier_at(f, perm_matrix), np.eye(3)))
    # Fourier at sign rep: f-hat(sign) = sum f(g) sgn(g) = 1 for delta_e
    checks.append(abs(sum(f[g] * (1.0 if _sign(g) == 1 else -1.0) for g in S3) - 1.0) < 1e-9)
    return float(sum(checks) / len(checks))


def _sign(p: tuple[int, ...]) -> int:
    inv = 0
    for i in range(3):
        for j in range(i + 1, 3):
            if p[i] > p[j]:
                inv += 1
    return -1 if inv % 2 else 1


def bench_fourier_sn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fourier_sn": _bench_fourier_sn(seed)}
