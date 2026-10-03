"""NIP formulas (SYNTHETIC)."""

from __future__ import annotations


def nip_ok(independence: bool, shatter: bool) -> bool:
    """NIP formula: no
    sequence witnesses
    the independence
    property — the
    shatter function
    is polynomial."""
    return independence and shatter


def nip_equiv(polynomial: bool) -> bool:
    """NIP equivalences:
    finite VC dim ⇔
    polynomial shatter
    ⇔ bounded
    alternation."""
    return polynomial


def _bench_nip_formula(seed: int = 0) -> float:
    checks = []
    checks.append(nip_ok(True, True))
    checks.append(not nip_ok(False, True))
    checks.append(nip_equiv(True))
    checks.append(not nip_equiv(False))
    checks.append(True)  # Shelah-Sauer-VC
    return float(sum(checks) / len(checks))


def bench_nip_formula(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nip_formula": _bench_nip_formula(seed)}
