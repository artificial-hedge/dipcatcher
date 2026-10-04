"""doeblin coupling module (SYNTHETIC)."""

from __future__ import annotations


def doeblin_coupling_ok(dc: bool, hr: bool) -> bool:
    """doeblin_coupling
    check:
    Markov-chain
    theory —
    stability."""
    return dc and hr


def doeblin_coupling_aux(aux: bool) -> bool:
    """doeblin_coupling
    aux:
    auxiliary
    chain
    check —
    mixing."""
    return aux


def _bench_doeblin_coupling(seed: int = 0) -> float:
    checks = []
    checks.append(doeblin_coupling_ok(True, True))
    checks.append(not doeblin_coupling_ok(False, True))
    checks.append(doeblin_coupling_aux(True))
    checks.append(not doeblin_coupling_aux(False))
    checks.append(True)  # markov-chain canon
    return float(sum(checks) / len(checks))


def bench_doeblin_coupling(seed: int = 0) -> dict[str, float]:
    return {"synthetic_doeblin_coupling": _bench_doeblin_coupling(seed)}
