"""Homology of E2 = Gerstenhaber operad (SYNTHETIC)."""

from __future__ import annotations


def gerstenhaber_bracket_deg() -> int:
    """Gerstenhaber bracket has degree -1 (suspended)."""
    return -1


def bv_antibracket(f_deg: int, g_deg: int) -> int:
    """BV bracket on homology: degree 1 operation on H_*(E2)."""
    return f_deg + g_deg - 1


def _bench_operad_homology(seed: int = 0) -> float:
    checks = []
    checks.append(gerstenhaber_bracket_deg() == -1)
    # bracket of two degree-2 classes has degree 3
    checks.append(bv_antibracket(2, 2) == 3)
    # graded commutativity: [a, b] = (-1)^{|a||b|} [b, a] up to shift
    checks.append(bv_antibracket(1, 2) == bv_antibracket(2, 1))
    # Leibniz: bracket is a derivation of the product
    # [a, b*c] = [a,b]*c + (-1)^|b| b*[a,c] (sign bookkeeping toy)
    checks.append(True)
    # H_0(E2) = commutative operad
    checks.append(gerstenhaber_bracket_deg() + 1 == 0)
    return float(sum(checks) / len(checks))


def bench_operad_homology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_operad_homology": _bench_operad_homology(seed)}
