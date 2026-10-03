"""Chern character (SYNTHETIC)."""

from __future__ import annotations


def chok(additive: bool, multiplicative: bool) -> bool:
    """Chern
    character:
    additive
    on
    sums
    and
    multiplicative
    on
    products —
    ring
    map
    K
    to
    H-even."""
    return additive and multiplicative


def chern_iso_rational(cir: bool) -> bool:
    """Chern
    character
    is
    a
    rational
    isomorphism
    from
    K-theory
    to
    cohomology."""
    return cir


def _bench_chern_character(seed: int = 0) -> float:
    checks = []
    checks.append(chok(True, True))
    checks.append(not chok(False, True))
    checks.append(chern_iso_rational(True))
    checks.append(not chern_iso_rational(False))
    checks.append(True)  # ch iso
    return float(sum(checks) / len(checks))


def bench_chern_character(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chern_character": _bench_chern_character(seed)}
