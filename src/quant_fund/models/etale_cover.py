"""Etale covers of curves: unramified + flat (SYNTHETIC)."""

from __future__ import annotations


def is_etale_at(ramification_index: int, residue_sep: bool, char: int) -> bool:
    """f: X -> Y etale at P iff e_P = 1 and residue extension separable
    (always in char 0)."""
    return ramification_index == 1 and residue_sep


def _bench_etale_cover(seed: int = 0) -> float:
    checks = []
    # identity map etale
    checks.append(is_etale_at(1, True, 0))
    # z -> z^2 ramified at 0 (e=2)
    checks.append(not is_etale_at(2, True, 0))
    # inseparable residue extension fails (char p Frobenius)
    checks.append(not is_etale_at(1, False, 3))
    # z -> z^n unramified at nonzero points in char 0
    checks.append(is_etale_at(1, True, 0))
    # Riemann-Hurwitz: 2g_X - 2 = d(2g_Y - 2) + sum(e_P - 1):
    # z -> z^2 on P^1 ramifies at 0 and infinity, each with e = 2:
    # -2 = 2*(-2) + 1 + 1
    checks.append(2 * 0 - 2 == 2 * (2 * 0 - 2) + (2 - 1) + (2 - 1))
    return float(sum(checks) / len(checks))


def bench_etale_cover(seed: int = 0) -> dict[str, float]:
    return {"synthetic_etale_cover": _bench_etale_cover(seed)}
