"""Arithmetical hierarchy: Σ_n/Π_n classification by quantifier alternation (SYNTHETIC)."""

from __future__ import annotations


def bounded_exists(pred, bound: int) -> bool:
    return any(pred(x) for x in range(bound))


def bounded_forall(pred, bound: int) -> bool:
    return all(pred(x) for x in range(bound))


def sigma1_holds(pred, bound: int) -> bool:
    """∃x pred(x) over finite bound."""
    return bounded_exists(pred, bound)


def pi1_holds(pred, bound: int) -> bool:
    return bounded_forall(pred, bound)


def sigma2_holds(pred, bound: int) -> bool:
    """∃x∀y pred(x,y)."""
    return bounded_exists(lambda x: bounded_forall(lambda y: pred(x, y), bound), bound)


def formula_level(form: tuple) -> int:
    """Count leading alternating blocks: ('exists',f)=1, ('forall','exists',f)=2..."""
    level = 0
    cur = form
    prev = None
    while isinstance(cur, tuple) and cur[0] in ("exists", "forall"):
        if cur[0] != prev:
            level += 1
            prev = cur[0]
        cur = cur[1]
    return level


def _bench_arith_hierarchy(seed: int = 0) -> float:
    checks = []
    checks.append(sigma1_holds(lambda x: x == 3, 10))
    checks.append(not sigma1_holds(lambda x: x == 10, 10))
    checks.append(pi1_holds(lambda x: x < 10, 10))
    checks.append(not pi1_holds(lambda x: x < 5, 10))
    checks.append(sigma2_holds(lambda x, y: x == 0, 5))  # x=0 works for all y
    checks.append(not sigma2_holds(lambda x, y: x < y, 5))  # no x < all y incl 0
    checks.append(formula_level(("exists", ("forall", ("exists", "P")))) == 3)
    checks.append(formula_level(("exists", ("exists", ("forall", "P")))) == 2)
    return float(sum(checks) / len(checks))


def bench_arith_hierarchy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_arith_hierarchy": _bench_arith_hierarchy(seed)}
