"""Karp–Luby FPRAS for DNF counting (SYNTHETIC bench)."""

from __future__ import annotations

import itertools
import random

Term = tuple[int, ...]


def term_satisfiers(term: Term, n: int) -> list[int]:
    """All assignments (bitmask ints) satisfying the term."""
    out = []
    for bits in itertools.product([0, 1], repeat=n):
        m = sum(b << (n - 1 - i) for i, b in enumerate(bits))
        ok = all(((m >> (n - 1 - (abs(lit) - 1))) & 1) == (1 if lit > 0 else 0) for lit in term)
        if ok:
            out.append(m)
    return out


def exact_dnf_count(terms: list[Term], n: int) -> int:
    union: set[int] = set()
    for t in terms:
        union |= set(term_satisfiers(t, n))
    return len(union)


def kl_count(terms: list[Term], n: int, eps: float, rng: random.Random) -> float:
    """Estimate |union of satisfiers| within (1±eps) w.h.p. — Karp–Luby."""
    sets = [set(term_satisfiers(t, n)) for t in terms]
    if not sets:
        return 0.0
    sizes = [len(s) for s in sets]
    m = sum(sizes)
    if m == 0:
        return 0.0
    t_needed = int(4 * len(terms) / (eps * eps)) + 1
    hits = 0
    for _ in range(t_needed):
        r = rng.randrange(m)
        acc = 0
        i = 0
        while acc + sizes[i] <= r:
            acc += sizes[i]
            i += 1
        a = sorted(sets[i])[rng.randrange(sizes[i])]
        # count terms the assignment satisfies; first satisfying term index
        first = min(j for j in range(len(terms)) if a in sets[j])
        if first == i:
            hits += 1
    return m * hits / t_needed


def _bench_fpras_dnf(seed: int = 0) -> float:
    rng = random.Random(20261231 + 1067)
    checks = []
    terms = [(1,), (2, -3)]
    checks.append(exact_dnf_count(terms, 3) == 5)
    checks.append(exact_dnf_count([(1, 2)], 3) == 2)
    # estimate within 20% on non-disjoint terms
    t = [(1,), (-1, 2)]
    exact = exact_dnf_count(t, 3)
    est = kl_count(t, 3, 0.15, rng)
    checks.append(abs(est - exact) <= 0.35 * exact)
    # empty DNF
    checks.append(kl_count([], 3, 0.1, rng) == 0.0)
    # full coverage: all assignments -> est close to 8
    est2 = kl_count([(1,), (-1,)], 3, 0.1, rng)
    checks.append(abs(est2 - 8) <= 0.5)
    return sum(checks) / len(checks)


def bench_fpras_dnf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fpras_dnf": _bench_fpras_dnf(seed)}
