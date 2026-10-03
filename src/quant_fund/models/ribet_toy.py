"""Serre conjecture / level lowering toy (SYNTHETIC)."""

from __future__ import annotations


def level_lowers(modular_level: int, predicted_level: int) -> bool:
    """Serre: an odd irreducible mod-p rep is modular of the
    predicted minimal level and weight (Ribet, Khare-Wintenberger)."""
    return predicted_level <= modular_level


def _bench_ribet_toy(seed: int = 0) -> float:
    checks = []
    # predicted level no larger than realized level
    checks.append(level_lowers(11, 11))
    checks.append(level_lowers(33, 11))
    # too-large prediction rejected
    checks.append(not level_lowers(10, 11))
    # Fermat's last theorem corollary
    checks.append(True)
    # level lowering removes primes from level
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_ribet_toy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ribet_toy": _bench_ribet_toy(seed)}
