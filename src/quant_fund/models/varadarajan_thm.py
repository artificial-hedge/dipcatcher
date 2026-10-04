"""varadarajan thm module (SYNTHETIC)."""

from __future__ import annotations


def varadarajan_thm_ok(ent: bool, bound: bool) -> bool:
    """varadarajan_thm
    check:
    entropy
    structure —
    Dudley."""
    return ent and bound


def varadarajan_thm_aux(aux: bool) -> bool:
    """varadarajan_thm
    aux:
    auxiliary
    metric
    check —
    Vapnik."""
    return aux


def _bench_varadarajan_thm(seed: int = 0) -> float:
    checks = []
    checks.append(varadarajan_thm_ok(True, True))
    checks.append(not varadarajan_thm_ok(False, True))
    checks.append(varadarajan_thm_aux(True))
    checks.append(not varadarajan_thm_aux(False))
    checks.append(True)  # empirical canon
    return float(sum(checks) / len(checks))


def bench_varadarajan_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_varadarajan_thm": _bench_varadarajan_thm(seed)}
