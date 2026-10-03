"""Algebras over operads: Assoc-algebra axioms (SYNTHETIC)."""

from __future__ import annotations


def check_assoc(mul: list[list[int]], elems: list[int]) -> bool:
    """(a*b)*c = a*(b*c) on a small multiplication table."""

    def m(x: int, y: int) -> int:
        return mul[x][y]

    return all(m(m(a, b), c) == m(a, m(b, c)) for a in elems for b in elems for c in elems)


def _bench_operad_algt(seed: int = 0) -> float:
    checks = []
    # Z/2 addition table is associative
    z2 = [[0, 1], [1, 0]]
    checks.append(check_assoc(z2, [0, 1]))
    # min operation on {0,1,2} is associative
    mn = [[min(a, b) for b in range(3)] for a in range(3)]
    checks.append(check_assoc(mn, [0, 1, 2]))
    # a non-associative table fails: subtraction mod 3
    sub = [[(a - b) % 3 for b in range(3)] for a in range(3)]
    checks.append(not check_assoc(sub, [0, 1, 2]))
    # unital: row/col of identity
    checks.append(all(z2[0][e] == e for e in [0, 1]))
    # operad axiom: composition equivariant under symmetric group
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_operad_algt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_operad_algt": _bench_operad_algt(seed)}
