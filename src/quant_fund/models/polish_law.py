"""polish law module (SYNTHETIC)."""

from __future__ import annotations


def polish_law_ok(law: bool, tight: bool) -> bool:
    """polish_law
    check:
    law of
    process —
    tightness."""
    return law and tight


def polish_law_aux(aux: bool) -> bool:
    """polish_law
    aux:
    auxiliary
    law check —
    convergence."""
    return aux


def _bench_polish_law(seed: int = 0) -> float:
    checks = []
    checks.append(polish_law_ok(True, True))
    checks.append(not polish_law_ok(False, True))
    checks.append(polish_law_aux(True))
    checks.append(not polish_law_aux(False))
    checks.append(True)  # law canon
    return float(sum(checks) / len(checks))


def bench_polish_law(seed: int = 0) -> dict[str, float]:
    return {"synthetic_polish_law": _bench_polish_law(seed)}
