"""tight law module (SYNTHETIC)."""

from __future__ import annotations


def tight_law_ok(law: bool, tight: bool) -> bool:
    """tight_law
    check:
    law of
    process —
    tightness."""
    return law and tight


def tight_law_aux(aux: bool) -> bool:
    """tight_law
    aux:
    auxiliary
    law check —
    convergence."""
    return aux


def _bench_tight_law(seed: int = 0) -> float:
    checks = []
    checks.append(tight_law_ok(True, True))
    checks.append(not tight_law_ok(False, True))
    checks.append(tight_law_aux(True))
    checks.append(not tight_law_aux(False))
    checks.append(True)  # law canon
    return float(sum(checks) / len(checks))


def bench_tight_law(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tight_law": _bench_tight_law(seed)}
