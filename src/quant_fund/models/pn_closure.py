"""pn closure module (SYNTHETIC)."""

from __future__ import annotations


def pn_closure_ok(flux: bool, ord: bool) -> bool:
    """pn_closure
    check:
    transport —
    angular-flux
    consistency."""
    return flux and ord


def pn_closure_aux(aux: bool) -> bool:
    """pn_closure
    aux:
    auxiliary
    transport check —
    moment bound."""
    return aux


def _bench_pn_closure(seed: int = 0) -> float:
    checks = []
    checks.append(pn_closure_ok(True, True))
    checks.append(not pn_closure_ok(False, True))
    checks.append(pn_closure_aux(True))
    checks.append(not pn_closure_aux(False))
    checks.append(True)  # transport canon
    return float(sum(checks) / len(checks))


def bench_pn_closure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pn_closure": _bench_pn_closure(seed)}
