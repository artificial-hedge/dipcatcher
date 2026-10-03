"""Herbrand models: least Herbrand model of a Horn program (SYNTHETIC)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def least_herbrand(facts: Iterable[str], rules: Sequence[tuple[str, str]]) -> set[str]:
    """Rules (body_atom -> head_atom) over a single 0-ary predicate toy:
    atoms are ground terms; iterate T_P until fixpoint."""
    model = set(facts)
    changed = True
    while changed:
        changed = False
        for body, head in rules:
            if body in model and head not in model:
                model.add(head)
                changed = True
    return model


def _bench_herbrand_model(seed: int = 0) -> float:
    checks = []
    # program: p(a); p(x) -> p(f(x)); q(a); p(y) -> q(g(y))
    # atoms as strings: 'pa','pfa','pffa', 'qa','qga', ...
    facts = {"p0", "q0"}
    rules = [("p0", "p1"), ("p1", "p2"), ("p2", "p3"), ("q0", "q1"), ("q1", "q2")]
    m = least_herbrand(facts, rules)
    checks.append(m == {"p0", "q0", "p1", "p2", "p3", "q1", "q2"})
    # empty facts -> least model empty
    checks.append(least_herbrand(set(), rules) == set())
    # unreachable rule bodies stay out
    checks.append("p4" not in m)
    # least model is minimal: removing q0 loses q1,q2
    m2 = least_herbrand({"p0"}, rules)
    checks.append(m2 == {"p0", "p1", "p2", "p3"})
    # fixpoint: applying rules again adds nothing
    m3 = least_herbrand(m, rules)
    checks.append(m3 == m)
    return float(sum(checks) / len(checks))


def bench_herbrand_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_herbrand_model": _bench_herbrand_model(seed)}
