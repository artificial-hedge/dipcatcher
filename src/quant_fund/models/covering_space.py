"""Graph covering maps and deck transformations (SYNTHETIC)."""

from __future__ import annotations


def covering_map(
    proj: dict[int, int], cover_adj: dict[int, set[int]], base_adj: dict[int, set[int]]
) -> bool:
    """proj: cover verts -> base verts is a covering iff for every vertex v,
    neighbors of v map bijectively onto neighbors of proj(v)."""
    for v, nbrs in cover_adj.items():
        b = proj[v]
        imgs = {proj[u] for u in nbrs}
        if imgs != base_adj.get(b, set()) or len(imgs) != len(nbrs):
            return False
    return True


def deck(proj: dict[int, int], cover_adj: dict[int, set[int]], perm: dict[int, int]) -> bool:
    """perm is a deck transformation iff it's a covering automorphism covering identity."""
    if any(proj.get(perm[v]) != proj.get(v) for v in perm):
        return False
    # graph automorphism
    for v, nbrs in cover_adj.items():
        if {perm[u] for u in nbrs} != cover_adj.get(perm[v], set()):
            return False
    return True


def _bench_covering_space(seed: int = 0) -> float:
    checks = []
    # double cover of circle C3 by C6: proj v mod 3
    base_adj = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
    cover_adj = {i: {(i + 1) % 6, (i - 1) % 6} for i in range(6)}
    proj = {i: i % 3 for i in range(6)}
    checks.append(covering_map(proj, cover_adj, base_adj))
    # deck transformation: rotation by 3
    perm = {i: (i + 3) % 6 for i in range(6)}
    checks.append(deck(proj, cover_adj, perm))
    # identity perm is deck
    checks.append(deck(proj, cover_adj, {i: i for i in range(6)}))
    # rotation by 1 is NOT deck (changes fiber)
    checks.append(not deck(proj, cover_adj, {i: (i + 1) % 6 for i in range(6)}))
    # not a covering: collapse two neighbors
    bad_proj = dict(proj)
    bad_proj[0] = 1
    checks.append(not covering_map(bad_proj, cover_adj, base_adj))
    return float(sum(checks) / len(checks))


def bench_covering_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_covering_space": _bench_covering_space(seed)}
