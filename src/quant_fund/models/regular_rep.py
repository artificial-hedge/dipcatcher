"""Regular representation: char = |G| at e else 0; decomposes as sum d_i chi_i (SYNTHETIC)."""

from __future__ import annotations

S3 = [(0, 1, 2), (1, 0, 2), (0, 2, 1), (2, 1, 0), (1, 2, 0), (2, 0, 1)]


def comp(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(p[q[i]] for i in range(3))


def reg_char(x: tuple[int, ...]) -> int:
    """Character of left regular rep: |G| if x=e else 0."""
    return sum(1 for g in S3 if comp(x, g) == g)


def decompose_mult(char_by_class: list[float], irrep: list[int], sizes: list[int]) -> int:
    return round(sum(s * a * b for s, a, b in zip(sizes, char_by_class, irrep, strict=True)) / 6)


def _bench_regular_rep(seed: int = 0) -> float:
    checks = []
    e = (0, 1, 2)
    t = (1, 0, 2)
    c = (1, 2, 0)
    checks.append(reg_char(e) == 6)
    checks.append(reg_char(t) == 0)
    checks.append(reg_char(c) == 0)
    # reg = 1*triv + 1*sign + 2*std: multiplicities = dims
    reg_by_class = [6.0, 0.0, 0.0]
    sizes = [1, 3, 2]
    checks.append(decompose_mult(reg_by_class, [1, 1, 1], sizes) == 1)
    checks.append(decompose_mult(reg_by_class, [1, -1, 1], sizes) == 1)
    checks.append(decompose_mult(reg_by_class, [2, 0, -1], sizes) == 2)
    return float(sum(checks) / len(checks))


def bench_regular_rep(seed: int = 0) -> dict[str, float]:
    return {"synthetic_regular_rep": _bench_regular_rep(seed)}
