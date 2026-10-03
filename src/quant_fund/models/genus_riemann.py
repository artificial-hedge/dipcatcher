"""Riemann-Hurwitz + genus-degree formulas for plane curves (SYNTHETIC)."""

from __future__ import annotations


def plane_genus(d: int) -> int:
    """Genus of smooth plane curve of degree d: (d-1)(d-2)/2."""
    return (d - 1) * (d - 2) // 2


def riemann_hurwitz(deg: int, g_target: int, branch: list[int]) -> int:
    """chi = deg*chi(target) - sum(e_p - 1); return source genus from total ramification."""
    chi = deg * (2 - 2 * g_target) - sum(b - 1 for b in branch)
    return (2 - chi) // 2


def elliptic_branch() -> list[int]:
    """Elliptic curve as double cover of P1 branched at 4 points, each e=2."""
    return [2, 2, 2, 2]


def _bench_genus_riemann(seed: int = 0) -> float:
    checks = []
    checks.append(plane_genus(1) == 0)  # line
    checks.append(plane_genus(2) == 0)  # conic
    checks.append(plane_genus(3) == 1)  # elliptic
    checks.append(plane_genus(4) == 3)  # quartic
    # Riemann-Hurwitz for elliptic double cover: chi = 2*2 - 4 = 0 -> g=1
    checks.append(riemann_hurwitz(2, 0, elliptic_branch()) == 1)
    # unramified double cover of genus-2: chi = 2*(-2) = -4 -> g=3
    checks.append(riemann_hurwitz(2, 2, []) == 3)
    return float(sum(checks) / len(checks))


def bench_genus_riemann(seed: int = 0) -> dict[str, float]:
    return {"synthetic_genus_riemann": _bench_genus_riemann(seed)}
