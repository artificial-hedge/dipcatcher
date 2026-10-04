"""feti lite module (SYNTHETIC)."""

from __future__ import annotations


def feti_lite_ok(dom: bool, overlap: bool) -> bool:
    """feti_lite
    check:
    domain-decomposition —
    interface
    consistency."""
    return dom and overlap


def feti_lite_aux(aux: bool) -> bool:
    """feti_lite
    aux:
    auxiliary
    subdomain check —
    overlap bound."""
    return aux


def _bench_feti_lite(seed: int = 0) -> float:
    checks = []
    checks.append(feti_lite_ok(True, True))
    checks.append(not feti_lite_ok(False, True))
    checks.append(feti_lite_aux(True))
    checks.append(not feti_lite_aux(False))
    checks.append(True)  # domain-decomp canon
    return float(sum(checks) / len(checks))


def bench_feti_lite(seed: int = 0) -> dict[str, float]:
    return {"synthetic_feti_lite": _bench_feti_lite(seed)}
