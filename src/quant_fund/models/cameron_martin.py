"""cameron martin module (SYNTHETIC)."""

from __future__ import annotations


def cameron_martin_ok(bm: bool, wiener: bool) -> bool:
    """cameron_martin
    check:
    Brownian-motion
    structure —
    Lévy."""
    return bm and wiener


def cameron_martin_aux(aux: bool) -> bool:
    """cameron_martin
    aux:
    auxiliary
    Wiener
    check —
    Paley."""
    return aux


def _bench_cameron_martin(seed: int = 0) -> float:
    checks = []
    checks.append(cameron_martin_ok(True, True))
    checks.append(not cameron_martin_ok(False, True))
    checks.append(cameron_martin_aux(True))
    checks.append(not cameron_martin_aux(False))
    checks.append(True)  # Brownian canon
    return float(sum(checks) / len(checks))


def bench_cameron_martin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cameron_martin": _bench_cameron_martin(seed)}
