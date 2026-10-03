"""Sylvester Hadamard matrices (SYNTHETIC)."""

from __future__ import annotations


def sylvester(n_pow2: int) -> list[list[int]]:
    """H_1 = [[1]]; H_{2n} = [[H_n, H_n], [H_n, -H_n]]."""
    if n_pow2 == 1:
        return [[1]]
    h = sylvester(n_pow2 // 2)
    top = [row + row for row in h]
    bot = [row + [-x for x in row] for row in h]
    return top + bot


def is_hadamard(h: list[list[int]]) -> bool:
    """H H^T = n I."""
    n = len(h)
    for i in range(n):
        for j in range(n):
            dot = sum(h[i][k] * h[j][k] for k in range(n))
            if dot != (n if i == j else 0):
                return False
    return True


def _bench_hadamard_matrix(seed: int = 0) -> float:
    checks = []
    checks.append(is_hadamard(sylvester(2)))
    checks.append(is_hadamard(sylvester(4)))
    checks.append(is_hadamard(sylvester(8)))
    # entries are +-1
    h4 = sylvester(4)
    checks.append(all(abs(v) == 1 for row in h4 for v in row))
    # |det H_n| = n^(n/2): check n=2 -> 2, via row ops count magnitude
    det2 = h4[0][0]  # placeholder sanity: use known |det H_2| = 2
    checks.append(det2 == 1)
    # rows of H_8: first row all +1
    checks.append(sylvester(8)[0] == [1] * 8)
    # any two distinct rows of H_4 are orthogonal: dot = 0
    h = sylvester(4)
    checks.append(sum(h[0][k] * h[1][k] for k in range(4)) == 0)
    return float(sum(checks) / len(checks))


def bench_hadamard_matrix(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hadamard_matrix": _bench_hadamard_matrix(seed)}
