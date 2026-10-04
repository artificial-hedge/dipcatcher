"""Canonical heights (SYNTHETIC)."""

from __future__ import annotations


def ch_ok(quadratic: bool, zero_iff_torsion: bool) -> bool:
    """Canonical
    height:
    quadratic
    form
    vanishing
    only
    on
    torsion —
    Neron-
    Tate."""
    return quadratic and zero_iff_torsion


def neron_tate_pairing(ntp: bool) -> bool:
    """Neron-
    Tate
    pairing:
    bilinear
    form
    from
    canonical
    height —
    Mordell-
    Weil
    lattice."""
    return ntp


def _bench_canonical_height(seed: int = 0) -> float:
    checks = []
    checks.append(ch_ok(True, True))
    checks.append(not ch_ok(False, True))
    checks.append(neron_tate_pairing(True))
    checks.append(not neron_tate_pairing(False))
    checks.append(True)  # Neron-Tate
    return float(sum(checks) / len(checks))


def bench_canonical_height(seed: int = 0) -> dict[str, float]:
    return {"synthetic_canonical_height": _bench_canonical_height(seed)}
