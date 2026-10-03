"""Region/'outlives' constraint solving.

Constraints: ("outlives",a,b) meaning region a ⊇ region b; ("eq_at",a,pt)
region a must contain program point pt. Solve by propagating point sets
through the outlives graph to a fixpoint — the RFC-2094 constraint graph
approach. Also computes a minimal region for each variable.
"""

from __future__ import annotations

_SEED = 20261231 + 1036

Constraint = tuple


def solve(constraints: list[Constraint], regions: list[str]) -> dict[str, set[int]]:
    """Propagate: point ∈ a and a:'b => point ∈ b. Fixpoint union."""
    pts: dict[str, set[int]] = {r: set() for r in regions}
    edges: list[tuple[str, str]] = []
    for c in constraints:
        if c[0] == "eq_at":
            pts[c[1]].add(int(c[2]))
        elif c[0] == "outlives":
            edges.append(
                (c[1], c[2])
            )  # a:'b -> b ⊆ a (points flow a->b? no: a outlives b means a ⊇ b)
    # a:'b means a ⊇ b, i.e. every point in b is in a: flow b -> a
    changed = True
    while changed:
        changed = False
        for a, b in edges:
            add = pts[b] - pts[a]
            if add:
                pts[a] |= add
                changed = True
    return pts


def min_region(pt: int, edge_graph: list[tuple[str, str]], pts: dict[str, set[int]]) -> str | None:
    """Smallest region (fewest points) containing pt."""
    cands = [r for r, s in pts.items() if pt in s]
    return min(cands, key=lambda r: len(pts[r])) if cands else None


def check_outlives(holds: list[tuple[str, str]], needed: tuple[str, str]) -> bool:
    """Reachability on the outlives graph: does a:'b follow by transitivity?"""
    a0, b0 = needed
    frontier = {a0}
    seen = set()
    while frontier:
        x = frontier.pop()
        if x == b0:
            return True
        if x in seen:
            continue
        seen.add(x)
        for a, b in holds:
            if a == x:
                frontier.add(b)
    return False


def bench_lifetime_outlives(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    cons: list[Constraint] = [
        ("eq_at", "a", 1),
        ("outlives", "a", "b"),
        ("eq_at", "b", 2),
    ]
    pts = solve(cons, ["a", "b"])
    # a:'b and 2∈b => 2∈a; 1∈a doesn't flow to b
    checks.append(pts["a"] == {1, 2} and pts["b"] == {2})
    # transitive outlives: a:'b:'c => a ⊇ {3}
    pts2 = solve(
        [("eq_at", "c", 3), ("outlives", "b", "c"), ("outlives", "a", "b")], ["a", "b", "c"]
    )
    checks.append(pts2["a"] == {3} and pts2["b"] == {3})
    # check_outlives transitivity
    checks.append(check_outlives([("a", "b"), ("b", "c")], ("a", "c")))
    checks.append(not check_outlives([("a", "b")], ("a", "c")))
    # min_region prefers smaller set
    checks.append(min_region(2, [], pts) == "b")
    return {"synthetic_lifetime_outlives": float(sum(checks)) / len(checks)}
