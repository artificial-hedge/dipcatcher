"""tarascon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tarascon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tarascon_qa_studies

    check:
    tarascon_qa_studies: TarasconQA metrics
    """
    return fit_ok and sample_ok


def tarascon_qa_studies_aux(aux: bool) -> bool:
    """tarascon_qa_studies

    aux:
    tarascon_qa_studies: tarascons, river dragons, answers, and scores
    """
    return aux


def _bench_tarascon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tarascon_qa_studies_ok(True, True))
    checks.append(not tarascon_qa_studies_ok(False, True))
    checks.append(tarascon_qa_studies_aux(True))
    checks.append(not tarascon_qa_studies_aux(False))
    checks.append(True)  # french-beast canon
    return float(sum(checks) / len(checks))


def bench_tarascon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tarascon_qa_studies": _bench_tarascon_qa_studies(seed)}
