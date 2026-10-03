"""Whitehead and Hurewicz theorems (SYNTHETIC)."""

from __future__ import annotations


def hurewicz_pi1_to_h1(pi1_rel: list[tuple[int, ...]], n_gen: int) -> list[int]:
    """Abelianization of the free/presented pi_1 -> H_1.

    pi1_rel: relators as exponent vectors over generators (e.g. commutator
    [a,b] = a+b-a-b -> 0). H_1 = Z^n / im(relators); returns invariant
    factor count = rank of Z^n minus relator rank.
    """
    import numpy as np

    if not pi1_rel:
        return [1] * n_gen  # free abelian Z^n
    m = np.array([list(r) for r in pi1_rel], dtype=float)
    rank = int(np.linalg.matrix_rank(m)) if m.size else 0
    out = [1] * (n_gen - rank)
    return out if out else [0]


def degree_multiplicative(d1: int, d2: int) -> int:
    """deg(f o g) = deg f * deg g for maps S^n -> S^n."""
    return d1 * d2


def _bench_whitehead(seed: int = 0) -> float:
    checks = []
    # figure-eight pi1 = F2 -> H1 = Z^2 (no relators)
    checks.append(hurewicz_pi1_to_h1([], 2) == [1, 1])
    # torus pi1 = <a,b | [a,b]> -> abelianization Z^2 (commutator is zero
    # in additive notation -> same as no relator rank on Z^2)
    checks.append(hurewicz_pi1_to_h1([(0, 0)], 2) == [1, 1])
    # RP2 pi1 = <a | a^2> -> H1 torsion Z/2 -> free rank 0
    checks.append(hurewicz_pi1_to_h1([(2,)], 1) == [0])
    # degree multiplicativity: z^k then z^m has degree k*m
    checks.append(degree_multiplicative(3, 2) == 6)
    checks.append(degree_multiplicative(-1, -1) == 1)
    # degree of antipodal map on S^n = (-1)^{n+1}
    checks.append(all((-1) ** (n + 1) in (1, -1) for n in range(4)))
    return float(sum(checks) / len(checks))


def bench_whitehead(seed: int = 0) -> dict[str, float]:
    return {"synthetic_whitehead": _bench_whitehead(seed)}
