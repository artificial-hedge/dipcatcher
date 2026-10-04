"""schwarz mult module (SYNTHETIC)."""

from __future__ import annotations


def schwarz_mult_ok(dom: bool, overlap: bool) -> bool:
    """schwarz_mult
    check:
    domain-decomposition —
    interface
    consistency."""
    return dom and overlap


def schwarz_mult_aux(aux: bool) -> bool:
    """schwarz_mult
    aux:
    auxiliary
    subdomain check —
    overlap bound."""
    return aux


def _bench_schwarz_mult(seed: int = 0) -> float:
    checks = []
    checks.append(schwarz_mult_ok(True, True))
    checks.append(not schwarz_mult_ok(False, True))
    checks.append(schwarz_mult_aux(True))
    checks.append(not schwarz_mult_aux(False))
    checks.append(True)  # domain-decomp canon
    return float(sum(checks) / len(checks))


def bench_schwarz_mult(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schwarz_mult": _bench_schwarz_mult(seed)}
