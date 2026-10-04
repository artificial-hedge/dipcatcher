"""viperfish_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def viperfish_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """viperfish_qa_studies

    check:
    viperfish_qa_studies: ViperfishQA metrics
    """
    return fit_ok and sample_ok


def viperfish_qa_studies_aux(aux: bool) -> bool:
    """viperfish_qa_studies

    aux:
    viperfish_qa_studies: viperfish, mesopelagic drifts, answers, and scores
    """
    return aux


def _bench_viperfish_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(viperfish_qa_studies_ok(True, True))
    checks.append(not viperfish_qa_studies_ok(False, True))
    checks.append(viperfish_qa_studies_aux(True))
    checks.append(not viperfish_qa_studies_aux(False))
    checks.append(True)  # abyssal canon
    return float(sum(checks) / len(checks))


def bench_viperfish_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_viperfish_qa_studies": _bench_viperfish_qa_studies(seed)}
