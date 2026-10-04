"""Frame correspondence for modal axioms (SYNTHETIC)."""

from __future__ import annotations


def reflexive(edges: set[tuple[int, int]], worlds: set[int]) -> bool:
    return all((w, w) in edges for w in worlds)


def transitive(edges: set[tuple[int, int]], worlds: set[int]) -> bool:
    return all((a, c) in edges for a, b in edges for b2, c in edges if b2 == b and a in worlds)


def symmetric(edges: set[tuple[int, int]], worlds: set[int]) -> bool:
    return all((b, a) in edges for a, b in edges)


def _bench_modal_completeness(seed: int = 0) -> float:
    checks = []
    w = {0, 1, 2}
    # reflexive frame validates T: {self loops}
    e_refl = {(0, 0), (1, 1), (2, 2), (0, 1), (1, 2)}
    checks.append(reflexive(e_refl, w))
    checks.append(not transitive(e_refl, w))
    # transitive frame validates 4
    e_tr = {(0, 1), (1, 2), (0, 2)}
    checks.append(transitive(e_tr, w))
    checks.append(not reflexive(e_tr, w))
    # symmetric validates B
    e_sym = {(0, 1), (1, 0)}
    checks.append(symmetric(e_sym, w))
    # S5 = equivalence: all three properties
    e_s5 = {(0, 0), (1, 1), (0, 1), (1, 0)}
    checks.append(reflexive(e_s5, {0, 1}))
    checks.append(transitive(e_s5, {0, 1}))
    checks.append(symmetric(e_s5, {0, 1}))
    # a non-equivalence fails S5
    checks.append(not symmetric({(0, 1)}, {0, 1}))
    return float(sum(checks) / len(checks))


def bench_modal_completeness(seed: int = 0) -> dict[str, float]:
    return {"synthetic_modal_completeness": _bench_modal_completeness(seed)}
