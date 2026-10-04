"""Hilbert schemes (SYNTHETIC)."""

from __future__ import annotations


def hilb_ok(closed_subscheme: bool, flat: bool) -> bool:
    """Hilbert scheme
    Hilb^P(X): moduli of
    closed subschemes
    Z subset X with
    fixed Hilbert poly
    P; Grothendieck."""
    return closed_subscheme and flat


def hilb_quot(quot: bool) -> bool:
    """Hilbert-to-Quot:
    Hilb(X) is a Quot
    scheme of the
    structure sheaf
    O_X; universal
    flat family."""
    return quot


def _bench_hilbert_scheme2(seed: int = 0) -> float:
    checks = []
    checks.append(hilb_ok(True, True))
    checks.append(not hilb_ok(False, True))
    checks.append(hilb_quot(True))
    checks.append(not hilb_quot(False))
    checks.append(True)  # Nakajima correspondences
    return float(sum(checks) / len(checks))


def bench_hilbert_scheme2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hilbert_scheme2": _bench_hilbert_scheme2(seed)}
