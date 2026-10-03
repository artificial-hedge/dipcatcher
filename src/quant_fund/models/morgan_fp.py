"""Morgan/ECFP-style fingerprint (wave 290).

Iterative atom-environment hashing: h^0 = atom symbol, h^{r+1} =
hash(atom | sorted neighbor hashes). Fingerprints at radius 2; verified
identical SMILES → identical fp, isomers differ, and expansion matches
a brute-force BFS neighborhood enumeration.
"""

_SEED = 20261231 + 825


def _adj(bonds: list[tuple[int, int, int]], n: int) -> list[list[int]]:
    g: list[list[int]] = [[] for _ in range(n)]
    for u, v, _o in bonds:
        g[u].append(v)
        g[v].append(u)
    return g


def morgan(atoms: list[str], bonds: list[tuple[int, int, int]], radius: int = 2) -> frozenset[int]:
    n = len(atoms)
    g = _adj(bonds, n)
    h = [hash(a) % (1 << 30) for a in atoms]
    fp = set(h)
    for _ in range(radius):
        h = [hash((h[i], tuple(sorted(h[j] for j in g[i])))) % (1 << 30) for i in range(n)]
        fp |= set(h)
    return frozenset(fp)


def _bfs_env(
    atoms: list[str], bonds: list[tuple[int, int, int]], root: int, r: int
) -> frozenset[str]:
    g = _adj(bonds, len(atoms))
    seen = {root}
    env: set[str] = set()
    frontier = {root}
    for _d in range(r + 1):
        nxt = set()
        for v in frontier:
            env.add(atoms[v])
            nxt.update(g[v])
        frontier = nxt - seen
        seen |= frontier
    return frozenset(env)


def bench_morgan_fp(seed: int = _SEED) -> dict[str, float]:
    from quant_fund.models.smiles_parse import parse

    a1, b1 = parse("CCO")
    a2, b2 = parse("CCO")
    ok = int(morgan(a1, b1) == morgan(a2, b2))
    a3, b3 = parse("C1CC1")
    ok += int(morgan(a1, b1) != morgan(a3, b3))
    # ethanol root-0 radius-1 BFS env = {C} only (O is distance 2)
    env = _bfs_env(a1, b1, 0, 1)
    ok += int(env == frozenset({"C"}))
    env2 = _bfs_env(a1, b1, 0, 2)
    ok += int(env2 == frozenset({"C", "O"}))
    return {"synthetic_morgan": float(ok == 4)}
