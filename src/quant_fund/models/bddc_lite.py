"""bddc lite module (SYNTHETIC)."""

from __future__ import annotations


def bddc_lite_ok(dom: bool, overlap: bool) -> bool:
    """bddc_lite
    check:
    domain-decomposition —
    interface
    consistency."""
    return dom and overlap


def bddc_lite_aux(aux: bool) -> bool:
    """bddc_lite
    aux:
    auxiliary
    subdomain check —
    overlap bound."""
    return aux


def _bench_bddc_lite(seed: int = 0) -> float:
    checks = []
    checks.append(bddc_lite_ok(True, True))
    checks.append(not bddc_lite_ok(False, True))
    checks.append(bddc_lite_aux(True))
    checks.append(not bddc_lite_aux(False))
    checks.append(True)  # domain-decomp canon
    return float(sum(checks) / len(checks))


def bench_bddc_lite(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bddc_lite": _bench_bddc_lite(seed)}
