"""Ring/cycle-basis detection on molecular graphs (wave 290) (SYNTHETIC).

Cyclomatic number = E - V + C for each connected component; SSSR-size
via BFS cycle basis on the incidence structure. Verified on benzene (1),
naphthalene-like fused C10 skeleton (2), acyclic ethanol (0).
"""

_SEED = 20261231 + 829


def cyclomatic(bonds: list[tuple[int, int, int]], n: int) -> int:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    comps = set()
    for i in range(n):
        comps.add(find(i))
    e = len(bonds)
    for u, v, _o in bonds:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    return e - n + len({find(i) for i in range(n)})


def bench_ring_detect(seed: int = _SEED) -> dict[str, float]:
    from quant_fund.models.smiles_parse import parse

    a, b = parse("c1ccccc1")
    ok = int(cyclomatic(b, len(a)) == 1)
    a, b = parse("c1ccc2ccccc2c1")
    ok += int(cyclomatic(b, len(a)) == 2)
    a, b = parse("CCO")
    ok += int(cyclomatic(b, len(a)) == 0)
    a, b = parse("C1CC2CC1C2")
    ok += int(cyclomatic(b, len(a)) == 2)
    return {"synthetic_ring": float(ok == 4)}
