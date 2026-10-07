"""Gysin / pullback-pushforward intersection maps (SYNTHIC) (SYNTHETIC)."""

from __future__ import annotations


def pullback_hyperplane(deg_f: int, mult: int = 1) -> int:
    """f^*H on a degree-d map f: X -> P^n pulls back to d*H (or d*mult)."""
    return deg_f * mult


def pushforward_degree(fiber_deg: int, target_pt_cls: int) -> int:
    """f_* of a point class on X mapping to P^n lands as fiber_deg * [pt]."""
    return fiber_deg * target_pt_cls


def proj_formula_deg(f: int, a_chow: int, b_pull: int) -> int:
    """Projection formula: f_*(a . f^* b) = f_*(a) . b ; on degrees:
    deg(f) * a_chow * b_pull."""
    return f * a_chow * b_pull


def _bench_gysin(seed: int = 0) -> float:
    checks = []
    # degree-2 cover f: P1 -> P1 pulls H back to 2H
    checks.append(pullback_hyperplane(2) == 2)
    # pushforward of a point on degree-3 cover -> 3[pt]
    checks.append(pushforward_degree(3, 1) == 3)
    # projection formula on a d-cover: f_*(pt . f^*H) = d * 1 * 1 = d
    checks.append(proj_formula_deg(3, 1, 1) == 3)
    # Veronese d=2 embedding P1 -> P2: image conic pulls O(1) to O(2)
    checks.append(pullback_hyperplane(2, 1) == 2)
    # finite map degree is multiplicative under composition
    checks.append(pullback_hyperplane(2, 3) == 6)
    # Gysin for inclusion i: Y -> X of codim c: i_*(i^* a) = a . [Y]
    # on P3: plane P2 includes; i_*(i^* h) = h . h = h^2 deg 1
    checks.append(proj_formula_deg(1, 1, 1) == 1)
    return float(sum(checks) / len(checks))


def bench_gysin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gysin": _bench_gysin(seed)}
