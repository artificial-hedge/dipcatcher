"""UOV-lite: unbalanced oil-vinegar signature over GF(2).

Central map F: n = v + o variables -> m = o quadratic equations, with no
oil x oil terms. Public map P = F . T for invertible mixing T. Signing picks
vinegar values, solves the remaining linear system in the oil variables.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_SEED = 20261231 + 941

_TERMS = tuple[tuple[int, int], ...]


def _terms(n: int) -> _TERMS:
    t = [(i, i) for i in range(n)]
    t += [(i, j) for i in range(n) for j in range(i + 1, n)]
    return tuple(t)


def _eval_quad(coef: np.ndarray, x: np.ndarray, n: int) -> int:
    acc = int(coef[-1])
    for k, (i, j) in enumerate(_terms(n)):
        if coef[k]:
            acc ^= int(x[i]) & int(x[j])
    return acc


def _rand_central(rng: np.random.Generator, v: int, o: int) -> np.ndarray:
    """m = o equations over n = v+o vars; entries in the term ordering."""
    n = v + o
    t = _terms(n)
    F = np.zeros((o, len(t) + 1), dtype=np.uint8)
    for k in range(o):
        for idx, (i, j) in enumerate(t):
            oil = i >= v and j >= v
            if not oil and rng.integers(0, 2):
                F[k, idx] = 1
        F[k, -1] = rng.integers(0, 2)
    # ensure each oil variable appears linearly somewhere per equation
    for k in range(o):
        for j in range(v, n):
            if not any(F[k, idx] for idx, (i, jj) in enumerate(t) if jj == j):
                i0 = int(rng.integers(0, v))
                F[k, t.index((min(i0, j), max(i0, j)))] = 1
    return F


def _gf2_inv(rng: np.random.Generator, n: int) -> np.ndarray:
    while True:
        m = (rng.integers(0, 2, (n, n))).astype(np.uint8)
        if _det(m):
            return m


def _det(a: np.ndarray) -> int:
    a = a.copy() % 2
    n = a.shape[0]
    for c in range(n):
        p = next((r for r in range(c, n) if a[r, c]), None)
        if p is None:
            return 0
        if p != c:
            a[[c, p]] = a[[p, c]]
        for r in range(c + 1, n):
            if a[r, c]:
                a[r] ^= a[c]
    return 1


def _solve(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    a = a.copy() % 2
    b = b.copy() % 2
    n = a.shape[0]
    aug = np.concatenate([a, b.reshape(-1, 1)], axis=1)
    for c in range(n):
        p = next((r for r in range(c, n) if aug[r, c]), None)
        if p is None:
            return None
        if p != c:
            aug[[c, p]] = aug[[p, c]]
        for r in range(n):
            if r != c and aug[r, c]:
                aug[r] ^= aug[c]
    return aug[:, n]


def _inverse(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    aug = np.concatenate([a % 2, np.eye(n, dtype=np.uint8)], axis=1)
    for c in range(n):
        p = next(r for r in range(c, n) if aug[r, c])
        if p != c:
            aug[[c, p]] = aug[[p, c]]
        for r in range(n):
            if r != c and aug[r, c]:
                aug[r] ^= aug[c]
    return aug[:, n:]


def _compose_public(F: np.ndarray, T: np.ndarray, n: int) -> np.ndarray:
    """P(x) = F(T x): expand quadratic composition term-by-term."""
    t = _terms(n)
    P = np.zeros((F.shape[0], len(t) + 1), dtype=np.uint8)
    # precompute each public term's linear-form expansion: T x
    for k in range(F.shape[0]):
        acc_lin = np.zeros(n, dtype=np.uint8)
        acc_quad: dict[tuple[int, int], int] = {}
        acc_const = int(F[k, -1])
        for idx, (i, j) in enumerate(t):
            if not F[k, idx]:
                continue
            ti = T[i]
            tj = T[j]
            # (ti . x)(tj . x) = sum_a ti[a]tj[a] x_a + sum_{a<b}(ti[a]tj[b]+ti[b]tj[a]) x_a x_b
            for a in range(n):
                if ti[a] and tj[a]:
                    acc_lin[a] ^= 1
            for a in range(n):
                for b in range(a + 1, n):
                    v = int(ti[a] & tj[b]) ^ int(ti[b] & tj[a])
                    if v:
                        key = (a, b)
                        acc_quad[key] = acc_quad.get(key, 0) ^ 1
        for idx, (i, j) in enumerate(t):
            v = acc_quad.get((i, j), 0)
            if i == j:
                v ^= int(acc_lin[i])
            P[k, idx] = v % 2
        P[k, -1] = acc_const % 2
    return P


def keygen(rng: np.random.Generator, v: int = 6, o: int = 6) -> dict[str, Any]:
    n = v + o
    F = _rand_central(rng, v, o)
    T = _gf2_inv(rng, n)
    P = _compose_public(F, T, n)
    return {"F": F, "T": T, "Ti": _inverse(T), "P": P, "v": v, "o": o, "n": n}


def public_eval(P: np.ndarray, x: np.ndarray, n: int) -> np.ndarray:
    return np.array([_eval_quad(P[k], x, n) for k in range(P.shape[0])], dtype=np.uint8)


def sign(
    kp: dict[str, Any], y: np.ndarray, rng: np.random.Generator, tries: int = 64
) -> np.ndarray | None:
    v, o, n = int(kp["v"]), int(kp["o"]), int(kp["n"])
    F, Ti = kp["F"], kp["Ti"]
    t = _terms(n)
    for _ in range(tries):
        vine = (rng.integers(0, 2, v)).astype(np.uint8)
        x0 = np.concatenate([vine, np.zeros(o, dtype=np.uint8)])
        A = np.zeros((o, o), dtype=np.uint8)
        rhs = y.copy() % 2
        for k in range(o):
            rhs[k] ^= _eval_quad(F[k], x0, n)
            for j in range(v, n):
                col = 0
                for i in range(v):
                    if vine[i]:
                        col ^= int(F[k, t.index((min(i, j), max(i, j)))])
                A[k, j - v] = col
        sol = _solve(A, rhs)
        if sol is None:
            continue
        z = np.concatenate([vine, sol]).astype(np.uint8)
        return np.asarray(Ti @ z % 2, dtype=np.uint8)
    return None


def bench_uov_sig(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    kp = keygen(rng)
    n = int(kp["n"])
    ok = 0
    trials = 40
    forged = 0
    for _ in range(trials):
        y = (rng.integers(0, 2, int(kp["o"]))).astype(np.uint8)
        s = sign(kp, y, rng)
        if s is not None and np.array_equal(public_eval(kp["P"], s, n), y):
            ok += 1
        sf = (rng.integers(0, 2, n)).astype(np.uint8)
        if np.array_equal(public_eval(kp["P"], sf, n), y):
            forged += 1
    score = 0.8 * (ok / trials) + 0.2 * (1.0 - forged / trials)
    return {"synthetic_uov_sig": score}
