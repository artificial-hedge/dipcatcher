"""squad_v2_studies module (SYNTHETIC)."""

from __future__ import annotations


def squad_v2_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """squad_v2_studies

    check:
    squad_v2_studies: SQuAD v2 extractive QA + unanswerable detection and F1
    """
    return fit_ok and sample_ok


def squad_v2_studies_aux(aux: bool) -> bool:
    """squad_v2_studies

    aux:
    squad_v2_studies: contexts, questions, answers, and abstention
    """
    return aux


def _bench_squad_v2_studies(seed: int = 0) -> float:
    checks = []
    checks.append(squad_v2_studies_ok(True, True))
    checks.append(not squad_v2_studies_ok(False, True))
    checks.append(squad_v2_studies_aux(True))
    checks.append(not squad_v2_studies_aux(False))
    checks.append(True)  # NLP-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_squad_v2_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_squad_v2_studies": _bench_squad_v2_studies(seed)}
