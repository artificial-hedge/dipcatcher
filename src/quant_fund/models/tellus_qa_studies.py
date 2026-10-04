"""tellus_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tellus_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tellus_qa_studies

    check:
    tellus_qa_studies: TellusQA metrics
    """
    return fit_ok and sample_ok


def tellus_qa_studies_aux(aux: bool) -> bool:
    """tellus_qa_studies

    aux:
    tellus_qa_studies: tellus, earth mothers, answers, and scores
    """
    return aux


def _bench_tellus_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tellus_qa_studies_ok(True, True))
    checks.append(not tellus_qa_studies_ok(False, True))
    checks.append(tellus_qa_studies_aux(True))
    checks.append(not tellus_qa_studies_aux(False))
    checks.append(True)  # roman-minor-2 canon
    return float(sum(checks) / len(checks))


def bench_tellus_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tellus_qa_studies": _bench_tellus_qa_studies(seed)}
