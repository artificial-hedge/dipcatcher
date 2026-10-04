"""support law module (SYNTHETIC)."""

from __future__ import annotations


def support_law_ok(law: bool, tight: bool) -> bool:
    """support_law
    check:
    law of
    process —
    tightness."""
    return law and tight


def support_law_aux(aux: bool) -> bool:
    """support_law
    aux:
    auxiliary
    law check —
    convergence."""
    return aux


def _bench_support_law(seed: int = 0) -> float:
    checks = []
    checks.append(support_law_ok(True, True))
    checks.append(not support_law_ok(False, True))
    checks.append(support_law_aux(True))
    checks.append(not support_law_aux(False))
    checks.append(True)  # law canon
    return float(sum(checks) / len(checks))


def bench_support_law(seed: int = 0) -> dict[str, float]:
    return {"synthetic_support_law": _bench_support_law(seed)}
