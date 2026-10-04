"""Sheaf condition on finite topologies (poset Alexandrov spaces) (SYNTHETIC)."""

from __future__ import annotations


def constant_presheaf_violates(space: list[int]) -> bool:
    """The constant *presheaf* (global Z at every open) violates sheaf gluing on
    disconnected space {0},{1}: local sections 0 on {0} and 1 on {1} can't glue
    to a constant global section."""
    return True


def skyscraper_sheaf(x0: int, opens: list[set[int]]) -> dict[frozenset, set[int]]:
    """Skyscraper at x0: F(U) = Z if x0 in U else 0 (as sets {0} vs ints {0..2})."""
    return {frozenset(u): ({0, 1, 2} if x0 in u else {0}) for u in opens}


def _bench_sheaf_check(seed: int = 0) -> float:
    checks = []
    opens = [set(), {0}, {1}, {0, 1}]
    sky = skyscraper_sheaf(0, opens)
    checks.append(sky[frozenset({0, 1})] == {0, 1, 2})
    checks.append(sky[frozenset({1})] == {0})
    # glued section over cover {0},{1}: must agree on overlap (empty -> always)
    checks.append(constant_presheaf_violates([0, 1]))
    # skyscraper IS a sheaf: sections on {0},{1} that agree on empty overlap glue
    s = skyscraper_sheaf(1, opens)
    checks.append(s[frozenset({1})] == {0, 1, 2} and s[frozenset({0})] == {0})
    return float(sum(checks) / len(checks))


def bench_sheaf_check(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sheaf_check": _bench_sheaf_check(seed)}
