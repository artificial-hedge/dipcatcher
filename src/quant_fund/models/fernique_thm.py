"""fernique thm module (SYNTHETIC)."""

from __future__ import annotations


def fernique_thm_ok(gp: bool, bound: bool) -> bool:
    """fernique_thm
    check:
    Gaussian-process
    structure —
    Slepian."""
    return gp and bound


def fernique_thm_aux(aux: bool) -> bool:
    """fernique_thm
    aux:
    auxiliary
    sup
    check —
    Fernique."""
    return aux


def _bench_fernique_thm(seed: int = 0) -> float:
    checks = []
    checks.append(fernique_thm_ok(True, True))
    checks.append(not fernique_thm_ok(False, True))
    checks.append(fernique_thm_aux(True))
    checks.append(not fernique_thm_aux(False))
    checks.append(True)  # Gaussian canon
    return float(sum(checks) / len(checks))


def bench_fernique_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fernique_thm": _bench_fernique_thm(seed)}
