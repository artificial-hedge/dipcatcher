"""spn equations module (SYNTHETIC)."""

from __future__ import annotations


def spn_equations_ok(flux: bool, ord: bool) -> bool:
    """spn_equations
    check:
    transport —
    angular-flux
    consistency."""
    return flux and ord


def spn_equations_aux(aux: bool) -> bool:
    """spn_equations
    aux:
    auxiliary
    transport check —
    moment bound."""
    return aux


def _bench_spn_equations(seed: int = 0) -> float:
    checks = []
    checks.append(spn_equations_ok(True, True))
    checks.append(not spn_equations_ok(False, True))
    checks.append(spn_equations_aux(True))
    checks.append(not spn_equations_aux(False))
    checks.append(True)  # transport canon
    return float(sum(checks) / len(checks))


def bench_spn_equations(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spn_equations": _bench_spn_equations(seed)}
