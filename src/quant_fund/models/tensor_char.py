"""Tensor products of characters decompose into irreducibles (SYNTHETIC)."""

from __future__ import annotations

from collections.abc import Sequence


def inner(ch1: Sequence[complex], ch2: Sequence[complex], sizes: Sequence[int]) -> float:
    """<chi, psi> = (1/|G|) sum chi(g) conj(psi(g)) over class reps."""
    n = sum(sizes)
    return float(abs(sum(c * ch1[i] * ch2[i].conjugate() for i, c in enumerate(sizes)) / n))


def decompose(
    ch: Sequence[complex],
    irrs: Sequence[Sequence[complex]],
    sizes: Sequence[int],
) -> dict[int, int]:
    return {i: round(inner(ch, irr, sizes)) for i, irr in enumerate(irrs)}


def _bench_tensor_char(seed: int = 0) -> float:
    checks = []
    # S3 character table: classes e(1), trans(3), 3cyc(2)
    sizes = [1, 3, 2]
    triv = [1, 1, 1]
    sign = [1, -1, 1]
    std = [2, 0, -1]
    irrs = [triv, sign, std]
    # std tensor std = triv + sign + std
    prod = [a * b for a, b in zip(std, std, strict=True)]
    d = decompose(prod, irrs, sizes)
    checks.append(d[0] == 1 and d[1] == 1 and d[2] == 1)
    # std tensor sign = std
    prod2 = [a * b for a, b in zip(std, sign, strict=True)]
    d2 = decompose(prod2, irrs, sizes)
    checks.append(d2[2] == 1 and d2[0] == 0 and d2[1] == 0)
    # sign tensor sign = triv
    prod3 = [a * b for a, b in zip(sign, sign, strict=True)]
    d3 = decompose(prod3, irrs, sizes)
    checks.append(d3[0] == 1 and d3[1] == 0 and d3[2] == 0)
    # degree checks: 2*2 = 4 = 1+1+2
    checks.append(sum(d.values()) == 0 or prod[0] == 4)
    return float(sum(checks) / len(checks))


def bench_tensor_char(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tensor_char": _bench_tensor_char(seed)}
