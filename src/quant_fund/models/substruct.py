"""Subgraph isomorphism on molecular graphs (wave 290) (SYNTHETIC).

VF2-lite backtracking: pattern atoms must match by symbol; bonds by
edge presence (order ignored). Benzene ring found in toluene, not in
ethanol.
"""

_SEED = 20261231 + 828


def _adj(bonds: list[tuple[int, int, int]], n: int) -> list[set[int]]:
    g: list[set[int]] = [set() for _ in range(n)]
    for u, v, _o in bonds:
        g[u].add(v)
        g[v].add(u)
    return g


def match(
    p_atoms: list[str],
    p_bonds: list[tuple[int, int, int]],
    t_atoms: list[str],
    t_bonds: list[tuple[int, int, int]],
) -> bool:
    pn, tn = len(p_atoms), len(t_atoms)
    if pn > tn:
        return False
    pg, tg = _adj(p_bonds, pn), _adj(t_bonds, tn)
    # order pattern by degree desc for pruning
    order = sorted(range(pn), key=lambda i: -len(pg[i]))
    mapping: dict[int, int] = {}
    used = [False] * tn

    def ok(pi: int, ti: int) -> bool:
        if p_atoms[pi].lower() != t_atoms[ti].lower():
            return False
        return all(tg[ti] & {mapping[pj]} for pj in pg[pi] if pj in mapping)

    def bt(k: int) -> bool:
        if k == pn:
            return True
        pi = order[k]
        for ti in range(tn):
            if used[ti] or not ok(pi, ti):
                continue
            # check mapped neighbors adjacency both ways
            if any(mapping[pj] not in tg[ti] for pj in pg[pi] if pj in mapping):
                continue
            mapping[pi] = ti
            used[ti] = True
            if bt(k + 1):
                return True
            del mapping[pi]
            used[ti] = False
        return False

    return bt(0)


def bench_substruct(seed: int = _SEED) -> dict[str, float]:
    from quant_fund.models.smiles_parse import parse

    benz = parse("c1ccccc1")
    tol = parse("Cc1ccccc1")
    eth = parse("CCO")
    ok = int(match(*benz, *tol))
    ok += int(not match(*benz, *eth))
    # OH group in ethanol: pattern "CO"
    ok += int(match(*parse("CO"), *eth))
    # ring of 5 not in benzene
    ok += int(not match(*parse("c1cccc1"), *tol))
    return {"synthetic_substruct": float(ok == 4)}
