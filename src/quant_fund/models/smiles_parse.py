"""SMILES parser (wave 290) (SYNTHETIC).

Minimal SMILES tokenizer → molecular graph: atoms C, N, O, S, P, H,
aromatic lowercase c/n, bonds =, #, ring digits 1-9, branches ().
Returns (atoms, bonds) lists; verified against hand-counted molecules.
"""

import re

_SEED = 20261231 + 824

_ATOM = re.compile(r"Cl|Br|[CNOSPFHIB]|[cnos]")


def parse(smiles: str) -> tuple[list[str], list[tuple[int, int, int]]]:
    atoms: list[str] = []
    bonds: list[tuple[int, int, int]] = []
    stack: list[int] = []
    rings: dict[str, tuple[int, int]] = {}
    i, cur, order = 0, -1, 1
    while i < len(smiles):
        m = _ATOM.match(smiles, i)
        if m:
            atoms.append(m.group(0))
            if cur >= 0:
                bonds.append((cur, len(atoms) - 1, order))
            cur = len(atoms) - 1
            order = 1
            i = m.end()
            continue
        c = smiles[i]
        if c == "=":
            order = 2
        elif c == "#":
            order = 3
        elif c == "(":
            stack.append(cur)
        elif c == ")":
            cur = stack.pop()
        elif c.isdigit():
            if c in rings:
                j, ro = rings.pop(c)
                bonds.append((j, cur, ro if order == 1 else order))
                order = 1
            else:
                rings[c] = (cur, order)
                order = 1
        i += 1
    return atoms, bonds


def bench_smiles_parse(seed: int = _SEED) -> dict[str, float]:
    ok = 0
    a, b = parse("CCO")
    ok += int(len(a) == 3 and len(b) == 2)
    a, b = parse("c1ccccc1")
    ok += int(len(a) == 6 and len(b) == 6)
    a, b = parse("CC(=O)O")
    ok += int(len(a) == 4 and b[1][2] == 2)
    a, b = parse("C1CC2CC1C2")
    ok += int(len(a) == 6 and len(b) == 7)
    a, b = parse("NCC(=O)O")
    ok += int(len(a) == 5 and sum(1 for t in b if t[2] == 2) == 1)
    return {"synthetic_smiles": float(ok == 5)}
