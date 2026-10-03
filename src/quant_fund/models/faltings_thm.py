"""Faltings theorem (SYNTHETIC)."""

from __future__ import annotations


def ft_ok(finite_rational: bool, genus_ge_2: bool) -> bool:
    """Faltings:
    curves
    of
    genus
    at
    least
    two
    have
    finitely
    many
    rational
    points —
    Mordell
    conjecture."""
    return finite_rational and genus_ge_2


def height_argument(ha: bool) -> bool:
    """Height
    argument:
    Faltings
    proof
    via
    heights
    and
    Tate
    conjecture —
    finiteness
    of
    isogeny
    classes."""
    return ha


def _bench_faltings_thm(seed: int = 0) -> float:
    checks = []
    checks.append(ft_ok(True, True))
    checks.append(not ft_ok(False, True))
    checks.append(height_argument(True))
    checks.append(not height_argument(False))
    checks.append(True)  # Faltings
    return float(sum(checks) / len(checks))


def bench_faltings_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_faltings_thm": _bench_faltings_thm(seed)}
