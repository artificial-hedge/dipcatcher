"""kea_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kea_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kea_qa_studies

    check:
    kea_qa_studies: KeaQA metrics
    """
    return fit_ok and sample_ok


def kea_qa_studies_aux(aux: bool) -> bool:
    """kea_qa_studies

    aux:
    kea_qa_studies: keas, alpine basins, answers, and scores
    """
    return aux


def _bench_kea_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kea_qa_studies_ok(True, True))
    checks.append(not kea_qa_studies_ok(False, True))
    checks.append(kea_qa_studies_aux(True))
    checks.append(not kea_qa_studies_aux(False))
    checks.append(True)  # parrot canon
    return float(sum(checks) / len(checks))


def bench_kea_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kea_qa_studies": _bench_kea_qa_studies(seed)}
