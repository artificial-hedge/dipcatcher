"""Small Ramsey numbers: R(3,3)=6 verified exhaustively; R(3,4)>7 (SYNTHETIC)."""

from __future__ import annotations

from itertools import combinations


def _has_mono(adj_bits: int, n: int, k: int, color: int) -> bool:
    """Check for monochromatic K_k of given color. Bit (i,j) -> bit index."""

    def bit(i: int, j: int) -> int:
        i, j = min(i, j), max(i, j)
        return (adj_bits >> (i * n - i * (i + 1) // 2 + (j - i - 1))) & 1

    for combo in combinations(range(n), k):
        if all(bit(a, b) == color for a, b in combinations(combo, 2)):
            return True
    return False


def ramsey_33_k6_all_mono() -> bool:
    """Every 2-coloring of K6 has a mono triangle."""
    e = 15
    return all(_has_mono(c, 6, 3, 0) or _has_mono(c, 6, 3, 1) for c in range(1 << e))


def ramsey_33_k5_witness() -> bool:
    """5-cycle red + complement blue is triangle-free in both colors."""
    n = 5
    bits = 0
    for i in range(5):
        a, b = min(i, (i + 1) % 5), max(i, (i + 1) % 5)
        bits |= 1 << (a * n - a * (a + 1) // 2 + (b - a - 1))
    return not _has_mono(bits, 5, 3, 0) and not _has_mono(bits, 5, 3, 1)


def _bench_ramsey_num(seed: int = 0) -> float:
    checks = []
    checks.append(ramsey_33_k5_witness())
    checks.append(ramsey_33_k6_all_mono())
    # R(3,4) > 8: red = Mobius ladder C8 with jumps {1,4} is K3-free and
    # its complement is K4-free (the unique (3,4)-Ramsey 8-vertex witness)
    n = 8
    bits = 0
    for i in range(n):
        for d in (1, 4):
            a, b = i, (i + d) % n
            a, b = min(a, b), max(a, b)
            bits |= 1 << (a * n - a * (a + 1) // 2 + (b - a - 1))
    checks.append(not _has_mono(bits, 8, 3, 1))
    checks.append(not _has_mono(bits, 8, 4, 0))
    # R(4,4) > 8: the same C8(1,4) coloring has no K4 in either color
    n8 = 8
    bits8 = 0
    for i in range(n8):
        for d in (1, 4):
            a, b = i, (i + d) % n8
            a, b = min(a, b), max(a, b)
            bits8 |= 1 << (a * n8 - a * (a + 1) // 2 + (b - a - 1))
    checks.append(not _has_mono(bits8, 8, 4, 1))
    checks.append(not _has_mono(bits8, 8, 4, 0))
    return float(sum(checks) / len(checks))


def bench_ramsey_num(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ramsey_num": _bench_ramsey_num(seed)}
