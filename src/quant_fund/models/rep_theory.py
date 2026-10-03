"""Finite-group representation theory: characters, orthogonality, S3 (SYNTHETIC)."""

from __future__ import annotations

import itertools

import numpy as np

Perm = tuple[int, ...]


def compose(p: Perm, q: Perm) -> Perm:
    return tuple(p[q[i]] for i in range(len(p)))


def perm_matrix(p: Perm) -> np.ndarray:
    n = len(p)
    m = np.zeros((n, n))
    for i in range(n):
        m[p[i], i] = 1.0
    return m


def character(rep, elems: list[Perm]) -> np.ndarray:
    return np.array([np.trace(rep(g)) for g in elems])


def inner(chi: np.ndarray, psi: np.ndarray, order: int) -> float:
    return float(np.vdot(chi, psi).real / order)


def _bench_rep_theory(seed: int = 0) -> float:
    checks = []
    s3 = list(itertools.permutations(range(3)))
    identity = (0, 1, 2)
    checks.append(compose((1, 0, 2), (0, 2, 1)) == (1, 2, 0))

    # standard representation of S3 on {sum=0} subspace is irreducible (dim 2)
    # use permutation rep chars: chi(g) = #fixed points
    def fp(p):
        return sum(1 for i in range(3) if p[i] == i)

    chi_perm = np.array([fp(g) for g in s3], dtype=float)
    chi_triv = np.ones(6)
    chi_sign = np.array([np.linalg.det(perm_matrix(g)) for g in s3])
    checks.append(inner(chi_triv, chi_triv, 6) == 1.0)
    checks.append(inner(chi_sign, chi_sign, 6) == 1.0)
    checks.append(inner(chi_triv, chi_sign, 6) == 0.0)
    # std char = perm - trivial; irreducible
    chi_std = chi_perm - chi_triv
    checks.append(inner(chi_std, chi_std, 6) == 1.0)
    checks.append(np.isclose(chi_std[s3.index(identity)], 2.0))
    # sum of squared dimensions = |G|  (1^2+1^2+2^2 = 6)
    checks.append(
        np.isclose(sum(c[s3.index(identity)] ** 2 for c in [chi_triv, chi_sign, chi_std]), 6.0)
    )
    return float(sum(checks) / len(checks))


def bench_rep_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rep_theory": _bench_rep_theory(seed)}
