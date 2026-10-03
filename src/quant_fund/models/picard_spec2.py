"""picard spec2 module (SYNTHETIC)."""

from __future__ import annotations


def picard_spec2_ok(chromatic: bool, height: bool) -> bool:
    """picard_spec2
    check:
    chromatic
    structure —
    height."""
    return chromatic and height


def picard_spec2_aux(aux: bool) -> bool:
    """picard_spec2
    aux:
    auxiliary
    chromatic
    check —
    periodicity."""
    return aux


def _bench_picard_spec2(seed: int = 0) -> float:
    checks = []
    checks.append(picard_spec2_ok(True, True))
    checks.append(not picard_spec2_ok(False, True))
    checks.append(picard_spec2_aux(True))
    checks.append(not picard_spec2_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_picard_spec2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_picard_spec2": _bench_picard_spec2(seed)}
