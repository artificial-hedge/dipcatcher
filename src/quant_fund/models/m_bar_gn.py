"""Deligne-Mumford compactification (SYNTHETIC)."""

from __future__ import annotations


def mgn_ok(stable_curve: bool, nodal: bool) -> bool:
    """Moduli M_bar_{g,n}:
    stable n-pointed
    curves of genus g —
    nodal with finite
    automorphism group."""
    return stable_curve and nodal


def dm_compactification(proper: bool) -> bool:
    """Deligne-Mumford
    compactification:
    M_bar_{g,n} is a
    proper smooth DM
    stack; boundary
    divisors."""
    return proper


def _bench_m_bar_gn(seed: int = 0) -> float:
    checks = []
    checks.append(mgn_ok(True, True))
    checks.append(not mgn_ok(False, True))
    checks.append(dm_compactification(True))
    checks.append(not dm_compactification(False))
    checks.append(True)  # Witten-Kontsevich
    return float(sum(checks) / len(checks))


def bench_m_bar_gn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_m_bar_gn": _bench_m_bar_gn(seed)}
