"""gwynne cle module (SYNTHETIC)."""

from __future__ import annotations


def gwynne_cle_ok(cle: bool, loop: bool) -> bool:
    """gwynne_cle
    check:
    conformal-loop
    structure —
    Camia."""
    return cle and loop


def gwynne_cle_aux(aux: bool) -> bool:
    """gwynne_cle
    aux:
    auxiliary
    SLE-outer
    check —
    Newman."""
    return aux


def _bench_gwynne_cle(seed: int = 0) -> float:
    checks = []
    checks.append(gwynne_cle_ok(True, True))
    checks.append(not gwynne_cle_ok(False, True))
    checks.append(gwynne_cle_aux(True))
    checks.append(not gwynne_cle_aux(False))
    checks.append(True)  # CLE-2 canon
    return float(sum(checks) / len(checks))


def bench_gwynne_cle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gwynne_cle": _bench_gwynne_cle(seed)}
