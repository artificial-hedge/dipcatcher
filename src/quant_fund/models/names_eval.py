"""P-names and evaluation under a generic (SYNTHETIC)."""

from __future__ import annotations

Cond = dict[int, int]
Name = list[tuple["Name | int", Cond]]  # set of (subname, condition) pairs


def eval_name(name: Name, g_elements: set[int]) -> set:
    """sigma^G = { tau^G : exists p in G with (tau, p) in sigma }.

    Conditions are represented as frozenset index into g_elements:
    we flatten each cond to a canonical id and check membership by
    testing whether some g-condition extends it (our G is a chain).
    """
    out: set = set()
    for sub, cond in name:
        # cond is a condition id into the generic's list of accepted conds
        if (
            sub in g_elements
            and frozenset(cond.items())
            in (frozenset() if cond == {} else {frozenset(cond.items())})
            or frozenset(cond.items()) in g_elements
        ):
            out.add(sub)
    return out


def check_name(name: Name, g_conds: list[Cond]) -> set[int]:
    """Evaluate a name whose subnames are literal ints."""
    out: set[int] = set()
    for sub, cond in name:
        if any(all(gc.get(k) == v for k, v in cond.items()) for gc in g_conds):
            out.add(int(sub))  # type: ignore[arg-type]
    return out


def _bench_names_eval(seed: int = 0) -> float:
    checks = []
    # name for the set {0,1} under Cohen generic G:
    # {(0, {}), (1, {}), (2, {0:1})} — 2 enters iff {0:1} in G
    name: Name = [(0, {}), (1, {}), (2, {0: 1})]
    g1 = [{}, {0: 1}]  # generic containing the condition {0:1}
    g2 = [{}, {0: 0}]  # generic not containing it
    checks.append(check_name(name, g1) == {0, 1, 2})
    checks.append(check_name(name, g2) == {0, 1})
    # canonical name for a ground-model set: check x_n = {(m,{}):m in n}
    x2: Name = [(0, {}), (1, {})]
    checks.append(check_name(x2, g2) == {0, 1})
    # empty name evaluates to empty set
    checks.append(check_name([], g1) == set())
    # name conditioned on incompatible conds picks the true branch only
    branch: Name = [(5, {0: 1}), (7, {0: 0})]
    checks.append(check_name(branch, g1) == {5})
    checks.append(check_name(branch, g2) == {7})
    return float(sum(checks) / len(checks))


def bench_names_eval(seed: int = 0) -> dict[str, float]:
    return {"synthetic_names_eval": _bench_names_eval(seed)}
