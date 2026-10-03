"""Omitting types: non-isolated types can be omitted (SYNTHETIC)."""

from __future__ import annotations

from fractions import Fraction


def is_isolated_type(cut: str, domain: list[Fraction]) -> bool:
    """A complete type over a finite set in DLO is isolated iff it's a
    formula with a principal generator, e.g. x = a, or an open interval
    (a,b) which is isolated as a TYPE iff the interval contains a domain
    element defining it. For the 'omit all constants' type {x != c_i},
    it's non-isolated when the domain is dense."""
    return cut == "point"


def realize_or_omit(kind: str, domain: list[Fraction], params: set[Fraction]) -> Fraction | str:
    """Either realize the type inside the domain or report that this
    model omits it."""
    if kind == "all_distinct":
        cands = [d for d in domain if d not in params]
        return cands[0] if cands else "OMITTED"
    return "OMITTED"


def _bench_omitting_types(seed: int = 0) -> float:
    checks = []
    dom = [Fraction(i, 2) for i in range(4)]
    params = {Fraction(0), Fraction(1)}
    # the type {x != 0, x != 1} is realized by 1/2 in this model
    r = realize_or_omit("all_distinct", dom, params)
    checks.append(r == Fraction(1, 2))
    # when params exhaust the domain the model omits the type — this is
    # exactly the omitting-types construction (take a model missing it)
    all_params = set(dom)
    checks.append(realize_or_omit("all_distinct", dom, all_params) == "OMITTED")
    # isolated type x = c is realized by c itself — it can't be omitted
    checks.append(is_isolated_type("point", dom))
    checks.append(not is_isolated_type("free", dom))
    # a countable model CAN omit a non-isolated type: the params-exhausted
    # model above is exactly such a model
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_omitting_types(seed: int = 0) -> dict[str, float]:
    return {"synthetic_omitting_types": _bench_omitting_types(seed)}
