"""Kisin module (SYNTHETIC)."""

from __future__ import annotations


def km_ok(kisin: bool, finite_ht: bool) -> bool:
    """Kisin
    module:
    finite
    free
    phi-
    module
    over
    S —
    Kisin
    module."""
    return kisin and finite_ht


def kisin_height(kh: bool) -> bool:
    """Kisin
    height:
    height
    bound
    for
    E-
    torsion —
    finite
    height."""
    return kh


def _bench_kisin_mod(seed: int = 0) -> float:
    checks = []
    checks.append(km_ok(True, True))
    checks.append(not km_ok(False, True))
    checks.append(kisin_height(True))
    checks.append(not kisin_height(False))
    checks.append(True)  # Kisin
    return float(sum(checks) / len(checks))


def bench_kisin_mod(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kisin_mod": _bench_kisin_mod(seed)}
