"""Permutation representations: character = fixed-point count, decomposition (SYNTHETIC)."""

from __future__ import annotations

Perm = tuple[int, ...]  # perm[i] = image of i


def fixed_points(p: Perm) -> int:
    return sum(1 for i, v in enumerate(p) if i == v)


def comp(p: Perm, q: Perm) -> Perm:
    return tuple(p[q[i]] for i in range(len(p)))


def perm_char(perms: list[tuple[int, int, int]]) -> list[int]:
    """Character of the permutation rep on each element."""
    return [fixed_points(p) for p in perms]


def multiplicity(char: list[int], irrep: list[int], class_sizes: list[int]) -> int:
    """<char, irrep> with class-multiplicity weights."""
    num = sum(s * a * b for s, a, b in zip(class_sizes, char, irrep, strict=True))
    den = sum(class_sizes)
    return round(num / den)


def _bench_perm_rep(seed: int = 0) -> float:
    checks = []
    s3 = [(0, 1, 2), (1, 0, 2), (0, 2, 1), (2, 1, 0), (1, 2, 0), (2, 0, 1)]
    chars = perm_char(s3)
    checks.append(chars == [3, 1, 1, 1, 0, 0])
    # class-averaged char vs irreps: <perm, triv> = (3 + 3*1)/6 = 1; <perm, sign> = (3-3)/6=0
    perm_by_class = [3, 1, 0]
    sizes = [1, 3, 2]
    checks.append(multiplicity(perm_by_class, [1, 1, 1], sizes) == 1)
    checks.append(multiplicity(perm_by_class, [1, -1, 1], sizes) == 0)
    checks.append(multiplicity(perm_by_class, [2, 0, -1], sizes) == 1)  # perm = triv + std
    checks.append(comp((1, 0, 2), (1, 0, 2)) == (0, 1, 2))
    checks.append(fixed_points(comp((1, 2, 0), (2, 0, 1))) == 3)  # 3cycle*3cycle = e
    return float(sum(checks) / len(checks))


def bench_perm_rep(seed: int = 0) -> dict[str, float]:
    return {"synthetic_perm_rep": _bench_perm_rep(seed)}
