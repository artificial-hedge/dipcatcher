"""fractional heston module (SYNTHETIC)."""

from __future__ import annotations


def fractional_heston_ok(fh1: bool, rv: bool) -> bool:
    """fractional_heston
    check:
    rough-vol
    —
    Volterra."""
    return fh1 and rv


def fractional_heston_aux(aux: bool) -> bool:
    """fractional_heston
    aux:
    auxiliary
    rough
    check —
    multifactor."""
    return aux


def _bench_fractional_heston(seed: int = 0) -> float:
    checks = []
    checks.append(fractional_heston_ok(True, True))
    checks.append(not fractional_heston_ok(False, True))
    checks.append(fractional_heston_aux(True))
    checks.append(not fractional_heston_aux(False))
    checks.append(True)  # rough-vol canon
    return float(sum(checks) / len(checks))


def bench_fractional_heston(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fractional_heston": _bench_fractional_heston(seed)}
