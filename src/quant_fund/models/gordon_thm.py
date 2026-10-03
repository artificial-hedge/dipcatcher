"""gordon thm module (SYNTHETIC)."""

from __future__ import annotations


def gordon_thm_ok(gp: bool, bound: bool) -> bool:
    """gordon_thm
    check:
    Gaussian-process
    structure —
    Slepian."""
    return gp and bound


def gordon_thm_aux(aux: bool) -> bool:
    """gordon_thm
    aux:
    auxiliary
    sup
    check —
    Fernique."""
    return aux


def _bench_gordon_thm(seed: int = 0) -> float:
    checks = []
    checks.append(gordon_thm_ok(True, True))
    checks.append(not gordon_thm_ok(False, True))
    checks.append(gordon_thm_aux(True))
    checks.append(not gordon_thm_aux(False))
    checks.append(True)  # Gaussian canon
    return float(sum(checks) / len(checks))


def bench_gordon_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gordon_thm": _bench_gordon_thm(seed)}
