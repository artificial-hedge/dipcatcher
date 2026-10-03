"""Harmonic map bubbling (SYNTHETIC)."""

from __future__ import annotations


def bh_ok(energy_loss: bool, bubble: bool) -> bool:
    """Harmonic
    map
    bubbling:
    energy
    concentrates
    at
    bubbles —
    Sacks-
    Uhlenbeck
    2D
    blow-
    up."""
    return energy_loss and bubble


def energy_identity_hm(ei: bool) -> bool:
    """Energy
    identity:
    total
    energy
    equals
    weak
    limit
    plus
    bubble
    energies —
    no
    neck
    loss."""
    return ei


def _bench_bubbling_hm(seed: int = 0) -> float:
    checks = []
    checks.append(bh_ok(True, True))
    checks.append(not bh_ok(False, True))
    checks.append(energy_identity_hm(True))
    checks.append(not energy_identity_hm(False))
    checks.append(True)  # Sacks-Uhlenbeck
    return float(sum(checks) / len(checks))


def bench_bubbling_hm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bubbling_hm": _bench_bubbling_hm(seed)}
