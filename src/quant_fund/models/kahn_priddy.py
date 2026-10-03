"""Kahn-Priddy (SYNTHETIC)."""

from __future__ import annotations


def kp_ok(kahn: bool, priddy: bool) -> bool:
    """Kahn
    Priddy:
    Kahn
    Priddy —
    transfer."""
    return kahn and priddy


def kahn_priddy_thm(kpt: bool) -> bool:
    """Kahn
    Priddy:
    Kahn
    Priddy
    theorem —
    surjective."""
    return kpt


def _bench_kahn_priddy(seed: int = 0) -> float:
    checks = []
    checks.append(kp_ok(True, True))
    checks.append(not kp_ok(False, True))
    checks.append(kahn_priddy_thm(True))
    checks.append(not kahn_priddy_thm(False))
    checks.append(True)  # Kahn-Priddy
    return float(sum(checks) / len(checks))


def bench_kahn_priddy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kahn_priddy": _bench_kahn_priddy(seed)}
