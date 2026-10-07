"""Naturality square checker (wave 289) (SYNTHETIC).

eta: F => G is natural iff G(f) o eta = eta o F(f) for every arrow
f — checked pointwise on endofunctors of a finite set Z6.
"""

_SEED = 20261231 + 820


def is_natural(
    f_map: dict[int, int], g_map: dict[int, int], eta: dict[int, int], arrows: list[int]
) -> bool:
    for _f in arrows:
        for a in range(6):
            lhs = g_map[eta[a]]
            rhs = eta[f_map[a]]
            if lhs != rhs:
                return False
    return True


def bench_nat_trans(seed: int = _SEED) -> dict[str, float]:
    # arrow on Z6: successor map; F and G both apply it.
    succ = {i: (i + 1) % 6 for i in range(6)}
    ident = {i: i for i in range(6)}
    # natural: eta = identity commutes with successor
    ok = is_natural(succ, succ, ident, [0])
    # also natural: eta = successor^2 (a power of the same map commutes)
    sq = {i: (i + 2) % 6 for i in range(6)}
    ok &= is_natural(succ, succ, sq, [0])
    # non-natural: eta = squaring map x^2 mod 6
    sqmap = {i: (i * i) % 6 for i in range(6)}
    bad = is_natural(succ, succ, sqmap, [0])
    return {"synthetic_nat_trans": float(ok and not bad)}
