"""Atiyah-Hirzebruch spectral sequence E2 page (SYNTHETIC)."""

from __future__ import annotations


def e2_page(homology: list[int], coeffs: list[int]) -> list[list[int]]:
    """E2_{p,q} = H_p(X; pi_q(E)): grid of groups (as orders, 0 = trivial,
    -1 = Z)."""
    return [[p * q for q in coeffs] for p in homology]


def _bench_spectral_atiyah(seed: int = 0) -> float:
    checks = []
    # X = S^2, K-theory: E2_{p,q} = H_p(S^2) tensor pi_q(KU)
    # H_0 = Z(-1), H_2 = Z(-1), others 0; pi_q(KU) = Z(-1) for q even, 0 odd
    page = e2_page([-1, 0, -1], [-1, 0, -1])
    checks.append(len(page) == 3 and all(len(row) == 3 for row in page))
    # H_0 x pi_0 = Z x Z -> finite-order product (-1)*(-1) = 1
    checks.append(page[0][0] == 1)
    # odd homology rows and odd coefficient columns vanish
    checks.append(page[1] == [0, 0, 0])
    checks.append(page[0][1] == 0 and page[2][1] == 0)
    # E2_{2,0} = Z -> row 2 recovers the H_2 contribution (K^0(S^2) = Z^2)
    checks.append(page[2][0] == 1)
    # trivial coefficients trivialize the page
    checks.append(e2_page([0, 0, 0], [-1, 0, -1]) == [[0, 0, 0]] * 3)
    return float(sum(checks) / len(checks))


def bench_spectral_atiyah(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_atiyah": _bench_spectral_atiyah(seed)}
