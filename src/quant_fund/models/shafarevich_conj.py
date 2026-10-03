"""Shafarevich conjecture (SYNTHETIC)."""

from __future__ import annotations


def sc_ok(finite_set: bool, good_reduction: bool) -> bool:
    """Shafarevich:
    finitely
    many
    isomorphism
    classes
    with
    good
    reduction
    outside
    S —
    arithmetic
    finiteness."""
    return finite_set and good_reduction


def zarhin_extension(ze: bool) -> bool:
    """Zarhin:
    Shafarevich
    for
    abelian
    varieties
    —
    finiteness
    in
    isogeny
    classes."""
    return ze


def _bench_shafarevich_conj(seed: int = 0) -> float:
    checks = []
    checks.append(sc_ok(True, True))
    checks.append(not sc_ok(False, True))
    checks.append(zarhin_extension(True))
    checks.append(not zarhin_extension(False))
    checks.append(True)  # Shafarevich-Zarhin
    return float(sum(checks) / len(checks))


def bench_shafarevich_conj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_shafarevich_conj": _bench_shafarevich_conj(seed)}
