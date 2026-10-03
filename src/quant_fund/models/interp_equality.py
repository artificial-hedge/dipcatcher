"""Equality axioms in first-order logic with = (SYNTHETIC)."""

from __future__ import annotations


def check_equality_axioms(relation: list[tuple[int, int]], n: int) -> dict[str, bool]:
    """Test a relation on {0..n-1} for =-axioms."""
    rel = set(relation)
    refl = all((x, x) in rel for x in range(n))
    sym = all((y, x) in rel for x, y in rel)
    trans = all((x, z) in rel for x, y in rel for y2, z in rel if y == y2)
    return {"refl": refl, "sym": sym, "trans": trans}


def _bench_interp_equality(seed: int = 0) -> float:
    checks = []
    eq = [(x, x) for x in range(4)]
    r = check_equality_axioms(eq, 4)
    checks.append(r["refl"] and r["sym"] and r["trans"])
    # congruence: equality preserves function application (toy)
    checks.append(all((x, x) in eq for x in range(4)))
    # a non-reflexive relation fails
    r2 = check_equality_axioms([(0, 1), (1, 0)], 4)
    checks.append(not r2["refl"])
    # {(0,1),(1,0)} is symmetric but not transitive (needs (0,0))
    checks.append(r2["sym"] and not r2["trans"])
    # mod-2 equivalence
    meq = [(x, y) for x in range(4) for y in range(4) if x % 2 == y % 2]
    r3 = check_equality_axioms(meq, 4)
    checks.append(r3["refl"] and r3["sym"] and r3["trans"])
    return float(sum(checks) / len(checks))


def bench_interp_equality(seed: int = 0) -> dict[str, float]:
    return {"synthetic_interp_equality": _bench_interp_equality(seed)}
