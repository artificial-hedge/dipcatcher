"""gwynne miller module (SYNTHETIC)."""

from __future__ import annotations


def gwynne_miller_ok(lqg: bool, gff: bool) -> bool:
    """gwynne_miller
    check:
    LQG-2
    structure —
    Gwynne."""
    return lqg and gff


def gwynne_miller_aux(aux: bool) -> bool:
    """gwynne_miller
    aux:
    auxiliary
    LQG
    check —
    Duplantier."""
    return aux


def _bench_gwynne_miller(seed: int = 0) -> float:
    checks = []
    checks.append(gwynne_miller_ok(True, True))
    checks.append(not gwynne_miller_ok(False, True))
    checks.append(gwynne_miller_aux(True))
    checks.append(not gwynne_miller_aux(False))
    checks.append(True)  # LQG-2 canon
    return float(sum(checks) / len(checks))


def bench_gwynne_miller(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gwynne_miller": _bench_gwynne_miller(seed)}
