"""Felsenstein pruning likelihood under Jukes-Cantor 69 (wave 284) (SYNTHETIC).

Two-taxon tree with root age tau and branch lengths t1,t2; JC69 transition
probs via (1/4)(1+3e^{-4t/3}) match. The pruning recursion equals the
closed-form two-leaf formula exactly.
"""

import numpy as np

_SEED = 20261231 + 792


def _p_same(t: float) -> float:
    return float(0.25 * (1 + 3 * np.exp(-4 * t / 3)))


def _p_diff(t: float) -> float:
    return float(0.25 * (1 - np.exp(-4 * t / 3)))


def lik_closed(l1: str, l2: str, t1: float, t2: float) -> float:
    # sum over root states: uniform root prior * P(l1|root,t1)*P(l2|root,t2)
    tot = 0.0
    for r in "ACGT":
        p1 = _p_same(t1) if r == l1 else _p_diff(t1)
        p2 = _p_same(t2) if r == l2 else _p_diff(t2)
        tot += 0.25 * p1 * p2
    return tot


def _prune(l1: str, l2: str, t1: float, t2: float) -> float:
    # generic recursion over explicit root states (what Felsenstein computes)
    bases = "ACGT"
    tot = 0.0
    for r in bases:
        p1v = [_p_same(t1) if b == l1 else _p_diff(t1) for b in bases]
        p2v = [_p_same(t2) if b == l2 else _p_diff(t2) for b in bases]
        tot += 0.25 * p1v[bases.index(r)] * p2v[bases.index(r)]
    return tot


def bench_jc69_lik(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(8):
        l1, l2 = rng.choice(list("ACGT"), 2)
        t1, t2 = rng.uniform(0.05, 1.5, 2)
        ok += int(abs(lik_closed(l1, l2, t1, t2) - _prune(l1, l2, t1, t2)) < 1e-12)
    # sanity: as t->inf, lik -> 1/16 for any pair
    v = lik_closed("A", "C", 50.0, 50.0)
    ok += int(abs(v - 1 / 16) < 1e-6)
    return {"synthetic_jc69": float(ok == 9)}
