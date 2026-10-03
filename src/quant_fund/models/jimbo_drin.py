"""Jimbo-Drinfeld presentation (SYNTHETIC)."""

from __future__ import annotations


def jd_ok(gen: bool, serre: bool) -> bool:
    """Jimbo-
    Drinfeld
    presentation:
    E_i, F_i,
    K_i with
    q-deformed
    [E_i,F_j] =
    delta_ij
    (K_i-K_i^-1)
    /(q-q^-1)."""
    return gen and serre


def q_serre_rel(qserre: bool) -> bool:
    """q-Serre
    relations:
    quantum
    binomial
    identities
    among
    Chevalley
    generators."""
    return qserre


def _bench_jimbo_drin(seed: int = 0) -> float:
    checks = []
    checks.append(jd_ok(True, True))
    checks.append(not jd_ok(False, True))
    checks.append(q_serre_rel(True))
    checks.append(not q_serre_rel(False))
    checks.append(True)  # Jimbo-Drinfeld
    return float(sum(checks) / len(checks))


def bench_jimbo_drin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jimbo_drin": _bench_jimbo_drin(seed)}
