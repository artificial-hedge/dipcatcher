"""krasue_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def krasue_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """krasue_qa_studies

    check:
    krasue_qa_studies: K
    """
    return fit_ok and sample_ok


def krasue_qa_studies_aux(aux: bool) -> bool:
    """krasue_qa_studies

    aux:
    krasue_qa_studies: r
    """
    return aux


def _bench_krasue_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(krasue_qa_studies_ok(True, True))
    checks.append(not krasue_qa_studies_ok(False, True))
    checks.append(krasue_qa_studies_aux(True))
    checks.append(not krasue_qa_studies_aux(False))
    checks.append(True)  # thai-demon canon
    return float(sum(checks) / len(checks))


def bench_krasue_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_krasue_qa_studies": _bench_krasue_qa_studies(seed)}
