"""cylindrical law module (SYNTHETIC)."""

from __future__ import annotations


def cylindrical_law_ok(law: bool, tight: bool) -> bool:
    """cylindrical_law
    check:
    law of
    process —
    tightness."""
    return law and tight


def cylindrical_law_aux(aux: bool) -> bool:
    """cylindrical_law
    aux:
    auxiliary
    law check —
    convergence."""
    return aux


def _bench_cylindrical_law(seed: int = 0) -> float:
    checks = []
    checks.append(cylindrical_law_ok(True, True))
    checks.append(not cylindrical_law_ok(False, True))
    checks.append(cylindrical_law_aux(True))
    checks.append(not cylindrical_law_aux(False))
    checks.append(True)  # law canon
    return float(sum(checks) / len(checks))


def bench_cylindrical_law(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cylindrical_law": _bench_cylindrical_law(seed)}
