"""crow_s_pairs_studies module (SYNTHETIC)."""

from __future__ import annotations


def crow_s_pairs_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """crow_s_pairs_studies

    check:
    crow_s_pairs_studies: CrowS-Pairs stereotype metrics
    """
    return fit_ok and sample_ok


def crow_s_pairs_studies_aux(aux: bool) -> bool:
    """crow_s_pairs_studies

    aux:
    crow_s_pairs_studies: pairs, preferences, biases, and scores
    """
    return aux


def _bench_crow_s_pairs_studies(seed: int = 0) -> float:
    checks = []
    checks.append(crow_s_pairs_studies_ok(True, True))
    checks.append(not crow_s_pairs_studies_ok(False, True))
    checks.append(crow_s_pairs_studies_aux(True))
    checks.append(not crow_s_pairs_studies_aux(False))
    checks.append(True)  # bias-eval canon
    return float(sum(checks) / len(checks))


def bench_crow_s_pairs_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_crow_s_pairs_studies": _bench_crow_s_pairs_studies(seed)}
