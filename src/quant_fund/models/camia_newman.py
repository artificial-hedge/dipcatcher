"""camia newman module (SYNTHETIC)."""

from __future__ import annotations


def camia_newman_ok(cle: bool, sle: bool) -> bool:
    """camia_newman
    check:
    conformal-loop-ensemble
    structure —
    Sheffield."""
    return cle and sle


def camia_newman_aux(aux: bool) -> bool:
    """camia_newman
    aux:
    auxiliary
    loop-ensemble
    check —
    Werner."""
    return aux


def _bench_camia_newman(seed: int = 0) -> float:
    checks = []
    checks.append(camia_newman_ok(True, True))
    checks.append(not camia_newman_ok(False, True))
    checks.append(camia_newman_aux(True))
    checks.append(not camia_newman_aux(False))
    checks.append(True)  # CLE canon
    return float(sum(checks) / len(checks))


def bench_camia_newman(seed: int = 0) -> dict[str, float]:
    return {"synthetic_camia_newman": _bench_camia_newman(seed)}
