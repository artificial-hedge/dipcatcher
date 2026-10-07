"""Sheaf gluing on a finite topological space cover (SYNTHETIC)."""

from __future__ import annotations

Section = dict[int, int]  # values on points of an open set


def agree_on_overlap(s1: Section, s2: Section, overlap: set[int]) -> bool:
    return all(s1[x] == s2[x] for x in overlap if x in s1 and x in s2) and all(
        x in s1 and x in s2 for x in overlap
    )


def glue(
    sections: list[Section], cover: list[set[int]], overlaps: list[tuple[int, int, set[int]]]
) -> Section | None:
    """Glue sections if pairwise-compatible on overlaps."""
    for i, j, ov in overlaps:
        if not agree_on_overlap(sections[i], sections[j], ov):
            return None
    out: Section = {}
    for s in sections:
        out.update(s)
    return out


def restriction(s: Section, u: set[int]) -> Section:
    return {x: v for x, v in s.items() if x in u}


def _bench_sheaf_gluing(seed: int = 0) -> float:
    checks = []
    u1, u2 = {0, 1}, {1, 2}
    cover = [u1, u2]
    overlaps = [(0, 1, {1})]
    s1, s2 = {0: 5, 1: 7}, {1: 7, 2: 9}
    g = glue([s1, s2], cover, overlaps)
    if not (g is not None):
        raise ValueError("g is not None")
    checks.append(g == {0: 5, 1: 7, 2: 9})
    checks.append(restriction(g, {0, 1}) == s1)
    # incompatible: no glue
    s3 = {1: 8, 2: 9}
    checks.append(glue([s1, s3], cover, overlaps) is None)
    # locally constant sheaf: gluing unique (identity axiom)
    g2 = glue([{0: 3, 1: 3}, {1: 3, 2: 3}], cover, overlaps)
    if not (g2 is not None):
        raise ValueError("g2 is not None")
    checks.append(g2 == {0: 3, 1: 3, 2: 3})
    # sheaf axiom: restrictions of a global section always agree
    checks.append(agree_on_overlap(restriction(g, u1), restriction(g, u2), {1}))
    # restriction is transitive
    checks.append(restriction(restriction(g, {0, 1, 2}), {0}) == restriction(g, {0}))
    return float(sum(checks) / len(checks))


def bench_sheaf_gluing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sheaf_gluing": _bench_sheaf_gluing(seed)}
