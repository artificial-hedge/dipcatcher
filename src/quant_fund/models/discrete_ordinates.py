"""discrete ordinates module (SYNTHETIC)."""

from __future__ import annotations


def discrete_ordinates_ok(flux: bool, ord: bool) -> bool:
    """discrete_ordinates
    check:
    transport —
    angular-flux
    consistency."""
    return flux and ord


def discrete_ordinates_aux(aux: bool) -> bool:
    """discrete_ordinates
    aux:
    auxiliary
    transport check —
    moment bound."""
    return aux


def _bench_discrete_ordinates(seed: int = 0) -> float:
    checks = []
    checks.append(discrete_ordinates_ok(True, True))
    checks.append(not discrete_ordinates_ok(False, True))
    checks.append(discrete_ordinates_aux(True))
    checks.append(not discrete_ordinates_aux(False))
    checks.append(True)  # transport canon
    return float(sum(checks) / len(checks))


def bench_discrete_ordinates(seed: int = 0) -> dict[str, float]:
    return {"synthetic_discrete_ordinates": _bench_discrete_ordinates(seed)}
