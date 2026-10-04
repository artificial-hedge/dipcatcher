"""Rainbow-lite: two-layer oil-vinegar signature over GF(2).

Layer 1: o1 equations in vinegars v and oils o1 (no o1^2 terms).
Layer 2: o2 equations in vinegars v+o1 and oils o2 (no o2^2 terms).
Public map P = F . T; signing solves two successive linear systems.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.uov_sig import (
    _compose_public,
    _eval_quad,
    _gf2_inv,
    _inverse,
    _solve,
    _terms,
    public_eval,
)

_SEED = 20261231 + 942


def _rand_layer(rng: np.random.Generator, n_vine: int, n_oil: int, n: int) -> np.ndarray:
    """n_oil polys: quadratic in vine vars (idx < n_vine), linear-cross into oils
    (idx in [n_vine, n_vine+n_oil)), no oil-oil terms. n total variables."""
    t = _terms(n)
    F = np.zeros((n_oil, len(t) + 1), dtype=np.uint8)
    for k in range(n_oil):
        for idx, (i, j) in enumerate(t):
            ok_ij = j < n_vine or (i < n_vine and n_vine <= j < n_vine + n_oil)
            if ok_ij and rng.integers(0, 2):
                F[k, idx] = 1
        F[k, -1] = rng.integers(0, 2)
    return F


def _layer_solve(F: np.ndarray, vine: np.ndarray, y: np.ndarray, n: int) -> np.ndarray | None:
    """Given vinegar assignment, solve linear system for oil variables."""
    n_oil = F.shape[0]
    n_vine = len(vine)
    t = _terms(n)
    x0 = np.zeros(n, dtype=np.uint8)
    x0[:n_vine] = vine
    A = np.zeros((n_oil, n_oil), dtype=np.uint8)
    rhs = y.copy() % 2
    for k in range(n_oil):
        rhs[k] ^= _eval_quad(F[k], x0, n)
        for j in range(n_vine, n_vine + n_oil):
            col = 0
            for i in range(n_vine):
                if vine[i]:
                    col ^= int(F[k, t.index((min(i, j), max(i, j)))])
            A[k, j - n_vine] = col
    return _solve(A, rhs)


def keygen(rng: np.random.Generator, v: int = 4, o1: int = 3, o2: int = 3) -> dict[str, Any]:
    n = v + o1 + o2
    F1 = _rand_layer(rng, v, o1, n)
    F2 = _rand_layer(rng, v + o1, o2, n)
    F = np.concatenate([F1, F2], axis=0)
    T = _gf2_inv(rng, n)
    P = _compose_public(F, T, n)
    return {
        "F1": F1,
        "F2": F2,
        "T": T,
        "Ti": _inverse(T),
        "P": P,
        "v": v,
        "o1": o1,
        "o2": o2,
        "n": n,
    }


def sign(
    kp: dict[str, Any], y: np.ndarray, rng: np.random.Generator, tries: int = 256
) -> np.ndarray | None:
    v, o1, n = int(kp["v"]), int(kp["o1"]), int(kp["n"])
    for _ in range(tries):
        vine = (rng.integers(0, 2, v)).astype(np.uint8)
        s1 = _layer_solve(kp["F1"], vine, y[:o1], n)
        if s1 is None:
            continue
        vine2 = np.concatenate([vine, s1])
        s2 = _layer_solve(kp["F2"], vine2, y[o1:], n)
        if s2 is None:
            continue
        z = np.concatenate([vine, s1, s2]).astype(np.uint8)
        return np.asarray(kp["Ti"] @ z % 2, dtype=np.uint8)
    return None


def bench_rainbow_sig(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    kp = keygen(rng)
    n, m = int(kp["n"]), int(kp["o1"]) + int(kp["o2"])
    ok = 0
    forged = 0
    trials = 40
    for _ in range(trials):
        y = (rng.integers(0, 2, m)).astype(np.uint8)
        s = sign(kp, y, rng)
        if s is not None and np.array_equal(public_eval(kp["P"], s, n), y):
            ok += 1
        sf = (rng.integers(0, 2, n)).astype(np.uint8)
        if np.array_equal(public_eval(kp["P"], sf, n), y):
            forged += 1
    forge_rate = forged / trials
    return {"synthetic_rainbow_sig": 0.8 * (ok / trials) + 0.2 * float(forge_rate <= 0.1)}
