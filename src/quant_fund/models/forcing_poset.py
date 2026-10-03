"""Cohen forcing poset: finite partial functions omega -> 2 (SYNTHETIC)."""

from __future__ import annotations

Cond = dict[int, int]  # finite partial function


def extends(p: Cond, q: Cond) -> bool:
    """p <= q in forcing order means p extends q (p superset of q)."""
    return all(k in p and p[k] == v for k, v in q.items())


def compatible(p: Cond, q: Cond) -> bool:
    """p, q compatible iff they agree on the common domain."""
    return all(p[k] == q[k] for k in p if k in q)


def union(p: Cond, q: Cond) -> Cond | None:
    """Common extension when compatible."""
    if not compatible(p, q):
        return None
    return {**p, **q}


def _bench_forcing_poset(seed: int = 0) -> float:
    checks = []
    p = {0: 1}
    q = {0: 1, 1: 0}
    r = {1: 0}
    # order axioms: reflexive, antisymmetric-ish, transitive
    checks.append(extends(p, p))
    checks.append(extends(q, p) and not extends(p, q))
    checks.append(extends(q, r) and extends(q, p) and extends({0: 1, 1: 0, 2: 1}, q))
    # compatibility: p={0:1} vs {0:0} incompatible; vs {1:0} compatible
    checks.append(not compatible(p, {0: 0}))
    checks.append(compatible(p, r))
    checks.append(union(p, r) == {0: 1, 1: 0})
    checks.append(union(p, {0: 0}) is None)
    # union is a common extension
    u = union(p, r)
    checks.append(u is not None and extends(u, p) and extends(u, r))
    return float(sum(checks) / len(checks))


def bench_forcing_poset(seed: int = 0) -> dict[str, float]:
    return {"synthetic_forcing_poset": _bench_forcing_poset(seed)}
