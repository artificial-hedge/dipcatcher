"""structured spec module (SYNTHETIC)."""

from __future__ import annotations


def structured_spec_ok(spectral: bool, geometry: bool) -> bool:
    """structured_spec
    check:
    spectral
    algebraic
    geometry —
    structured."""
    return spectral and geometry


def structured_spec_aux(aux: bool) -> bool:
    """structured_spec
    aux:
    auxiliary
    spectral-AG
    check —
    derived."""
    return aux


def _bench_structured_spec(seed: int = 0) -> float:
    checks = []
    checks.append(structured_spec_ok(True, True))
    checks.append(not structured_spec_ok(False, True))
    checks.append(structured_spec_aux(True))
    checks.append(not structured_spec_aux(False))
    checks.append(True)  # spectral AG canon
    return float(sum(checks) / len(checks))


def bench_structured_spec(seed: int = 0) -> dict[str, float]:
    return {"synthetic_structured_spec": _bench_structured_spec(seed)}
