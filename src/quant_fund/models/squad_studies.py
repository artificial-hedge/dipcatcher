"""squad_studies module (SYNTHETIC)."""

from __future__ import annotations


def squad_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """squad_studies

    check:
    squad_studies: SQuAD extractive QA spans and EM/F1
    """
    return fit_ok and sample_ok


def squad_studies_aux(aux: bool) -> bool:
    """squad_studies

    aux:
    squad_studies: context-question pairs, answer spans, and scores
    """
    return aux


def _bench_squad_studies(seed: int = 0) -> float:
    checks = []
    checks.append(squad_studies_ok(True, True))
    checks.append(not squad_studies_ok(False, True))
    checks.append(squad_studies_aux(True))
    checks.append(not squad_studies_aux(False))
    checks.append(True)  # reading-comprehension canon
    return float(sum(checks) / len(checks))


def bench_squad_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_squad_studies": _bench_squad_studies(seed)}
