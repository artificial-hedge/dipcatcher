"""Cocartesian fibration (SYNTHETIC)."""

from __future__ import annotations


def cc_ok(cocartesian: bool, lift: bool) -> bool:
    """Cocartesian:
    cocartesian
    fibration
    —
    Lurie
    cocart."""
    return cocartesian and lift


def cocart_lift(cl: bool) -> bool:
    """Cocartesian
    lift:
    cocartesian
    lifting
    morphisms —
    Lurie
    lift."""
    return cl


def _bench_cocartesian(seed: int = 0) -> float:
    checks = []
    checks.append(cc_ok(True, True))
    checks.append(not cc_ok(False, True))
    checks.append(cocart_lift(True))
    checks.append(not cocart_lift(False))
    checks.append(True)  # Lurie
    return float(sum(checks) / len(checks))


def bench_cocartesian(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cocartesian": _bench_cocartesian(seed)}
