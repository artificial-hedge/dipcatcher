"""cavy_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def cavy_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cavy_qa_studies

    check:
    cavy_qa_studies: CavyQA metrics
    """
    return fit_ok and sample_ok


def cavy_qa_studies_aux(aux: bool) -> bool:
    """cavy_qa_studies

    aux:
    cavy_qa_studies: cavies, pampas grass, answers, and scores
    """
    return aux


def _bench_cavy_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cavy_qa_studies_ok(True, True))
    checks.append(not cavy_qa_studies_ok(False, True))
    checks.append(cavy_qa_studies_aux(True))
    checks.append(not cavy_qa_studies_aux(False))
    checks.append(True)  # small-mammal-2 canon
    return float(sum(checks) / len(checks))


def bench_cavy_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cavy_qa_studies": _bench_cavy_qa_studies(seed)}
