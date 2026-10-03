"""Rank-1 constraint system satisfiability over a prime field.

R1CS: matrices A,B,C over Fp; witness w satisfies iff (A w) ⊙ (B w) = C w
rowwise. Helpers build common gadgets (mul, linear combination, boolean)
and verify a full assignment.
"""

from __future__ import annotations

_SEED = 20261231 + 1047

P = 97  # small prime; benches only need smallness


def mat_vec(m: list[list[int]], w: list[int], p: int = P) -> list[int]:
    return [sum(a * x for a, x in zip(row, w, strict=True)) % p for row in m]


def satisfies(
    a: list[list[int]], b: list[list[int]], c: list[list[int]], w: list[int], p: int = P
) -> bool:
    aw = mat_vec(a, w, p)
    bw = mat_vec(b, w, p)
    cw = mat_vec(c, w, p)
    return all(x * y % p == z for x, y, z in zip(aw, bw, cw, strict=True))


def mul_gadget(
    x: int, y: int, z: int, p: int = P
) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[int]]:
    """w = [1, x, y, z] with z = x*y."""
    a = [[0, 1, 0, 0]]
    b = [[0, 0, 1, 0]]
    c = [[0, 0, 0, 1]]
    return a, b, c, [1, x % p, y % p, (x * y) % p]


def bench_r1cs_check(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # x*y=z: 3*4=12
    a, b, c, w = mul_gadget(3, 4, 12)
    checks.append(satisfies(a, b, c, w))
    # wrong witness: claims 3*4=13
    a, b, c, w = mul_gadget(3, 4, 12)
    w[3] = 13
    checks.append(not satisfies(a, b, c, w))
    # (x+1)*(x+1) = x^2+2x+1 at x=5 -> 36
    a2 = [[1, 1, 0, 0]]
    b2 = [[1, 1, 0, 0]]
    c2 = [[0, 0, 0, 1]]
    w2 = [1, 5, 0, 36]
    checks.append(satisfies(a2, b2, c2, w2))
    # wrap-around: 50*60 mod 97 = 90
    a3, b3, c3, w3 = mul_gadget(50, 60, (50 * 60) % 97)
    checks.append(satisfies(a3, b3, c3, w3))
    # two-row: x*y=z and u*v=t both hold
    a4 = [[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    b4 = [[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]]
    c4 = [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]]
    w4 = [1, 3, 4, 5, 6, 30]
    # row1 claims 3*4 = w[5]=30 -> unsat even though row2 holds
    checks.append(not satisfies(a4, b4, c4, w4))
    a5 = [[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]]
    b5 = [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0]]
    c5 = [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]]
    w5 = [1, 3, 4, 12, 5, 6, 30]
    checks.append(satisfies(a5, b5, c5, w5))
    return {"synthetic_r1cs_check": float(sum(checks)) / len(checks)}
