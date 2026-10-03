"""K_0 of coherent sheaves / vector bundles on toy varieties (SYNTHETIC)."""

from __future__ import annotations


def rank_map(k0_elt: tuple[int, dict[str, int]]) -> int:
    """K_0 element = (virtual rank, Chern data); rank is additive."""
    return k0_elt[0]


def chern_character_line_bundle(c1_num: int, c1_den: int = 1) -> list[float]:
    """ch(L) = exp(c1(L)) = 1 + c1 + c1^2/2 + ... truncated at degree 2."""
    c1 = c1_num / c1_den
    return [1.0, c1, c1 * c1 / 2]


def riemann_roch_curve(chi0: int, deg: int, genus: int) -> int:
    """chi(E) = deg(E) + rank(E) * (1 - g) on a curve."""
    return deg + chi0 * (1 - genus)


def _bench_grothendieck_grp(seed: int = 0) -> float:
    checks = []
    # ch(L^k) has c1 = k*c1(L)
    checks.append(chern_character_line_bundle(3)[1] == 3.0)
    # K0(P1): structure sheaf + point class generate; chi(O(d)) = d+1
    checks.append(riemann_roch_curve(1, 5, 0) == 6)
    checks.append(riemann_roch_curve(1, -2, 0) == -1)
    # elliptic curve g=1: chi = deg
    checks.append(riemann_roch_curve(1, 7, 1) == 7)
    # genus 2: chi(O(D)) = deg - 1
    checks.append(riemann_roch_curve(1, 4, 2) == 3)
    # additivity of rank in K0: [E]+[F] rank adds
    checks.append(rank_map((2, {})) + rank_map((3, {})) == 5)
    return float(sum(checks) / len(checks))


def bench_grothendieck_grp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_grothendieck_grp": _bench_grothendieck_grp(seed)}
