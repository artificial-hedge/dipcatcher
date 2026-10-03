"""Molecular descriptors (wave 290).

Molecular weight, hydrogen-bond donors/acceptors (N/O counts),
rotatable bonds (non-ring single bonds with degree>1 endpoints) —
verified against hand-computed values on small molecules.
"""

_SEED = 20261231 + 827

_MASS = {
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
    "P": 30.974,
    "H": 1.008,
    "Cl": 35.45,
    "Br": 79.904,
    "F": 18.998,
    "I": 126.90,
    "B": 10.81,
}


def mol_weight(atoms: list[str]) -> float:
    return sum(_MASS.get(a.capitalize(), 12.011) for a in atoms)


def h_donors(atoms: list[str]) -> int:
    return sum(1 for a in atoms if a in ("N", "O", "n"))


def h_acceptors(atoms: list[str]) -> int:
    return sum(1 for a in atoms if a in ("N", "O", "S", "n", "o"))


def rotatable(atoms: list[str], bonds: list[tuple[int, int, int]]) -> int:
    n = len(atoms)
    deg = [0] * n
    in_ring = [False] * len(bonds)
    # edge in ring iff removing it still connects endpoints (cycle check per edge)
    g: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for ei, (u, v, _o) in enumerate(bonds):
        g[u].append((v, ei))
        g[v].append((u, ei))
    for ei, (u, v, _o) in enumerate(bonds):
        seen = {u}
        stack = [u]
        while stack:
            x = stack.pop()
            for y, e2 in g[x]:
                if e2 == ei:
                    continue
                if y == v:
                    in_ring[ei] = True
                    stack = []
                    break
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
    deg = [len({j for j, _ in g[i]}) for i in range(n)]
    count = 0
    for ei, (u, v, o) in enumerate(bonds):
        if o == 1 and not in_ring[ei] and deg[u] > 1 and deg[v] > 1:
            count += 1
    return count


def bench_mol_descriptors(seed: int = _SEED) -> dict[str, float]:
    from quant_fund.models.smiles_parse import parse

    ok = 0
    a, b = parse("CCO")
    ok += int(abs(mol_weight(a) - 40.021) < 0.01)
    ok += int(h_donors(a) == 1 and h_acceptors(a) == 1)
    a, b = parse("CCCC")
    ok += int(rotatable(a, b) == 1)
    a, b = parse("CC(=O)O")
    ok += int(rotatable(a, b) == 0 and abs(mol_weight(a) - 56.02) < 0.01)
    a, b = parse("c1ccccc1")
    ok += int(rotatable(a, b) == 0 and abs(mol_weight(a) - 72.066) < 0.01)
    return {"synthetic_mol_descriptors": float(ok == 5)}
