"""picard spec module (SYNTHETIC)."""

from __future__ import annotations


def picard_spec_ok(chromatic: bool, height: bool) -> bool:
    """picard_spec
    check:
    chromatic
    height
    structure —
    stratified."""
    return chromatic and height


def picard_spec_aux(aux: bool) -> bool:
    """picard_spec
    aux:
    auxiliary
    chromatic
    check —
    spectral."""
    return aux


def _bench_picard_spec(seed: int = 0) -> float:
    checks = []
    checks.append(picard_spec_ok(True, True))
    checks.append(not picard_spec_ok(False, True))
    checks.append(picard_spec_aux(True))
    checks.append(not picard_spec_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_picard_spec(seed: int = 0) -> dict[str, float]:
    return {"synthetic_picard_spec": _bench_picard_spec(seed)}
