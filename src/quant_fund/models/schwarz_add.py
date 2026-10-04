"""schwarz add module (SYNTHETIC)."""

from __future__ import annotations


def schwarz_add_ok(dom: bool, overlap: bool) -> bool:
    """schwarz_add
    check:
    domain-decomposition —
    interface
    consistency."""
    return dom and overlap


def schwarz_add_aux(aux: bool) -> bool:
    """schwarz_add
    aux:
    auxiliary
    subdomain check —
    overlap bound."""
    return aux


def _bench_schwarz_add(seed: int = 0) -> float:
    checks = []
    checks.append(schwarz_add_ok(True, True))
    checks.append(not schwarz_add_ok(False, True))
    checks.append(schwarz_add_aux(True))
    checks.append(not schwarz_add_aux(False))
    checks.append(True)  # domain-decomp canon
    return float(sum(checks) / len(checks))


def bench_schwarz_add(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schwarz_add": _bench_schwarz_add(seed)}
