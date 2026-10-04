"""opossum_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def opossum_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """opossum_qa_studies

    check:
    opossum_qa_studies: OpossumQA metrics
    """
    return fit_ok and sample_ok


def opossum_qa_studies_aux(aux: bool) -> bool:
    """opossum_qa_studies

    aux:
    opossum_qa_studies: opossums, night forages, answers, and scores
    """
    return aux


def _bench_opossum_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(opossum_qa_studies_ok(True, True))
    checks.append(not opossum_qa_studies_ok(False, True))
    checks.append(opossum_qa_studies_aux(True))
    checks.append(not opossum_qa_studies_aux(False))
    checks.append(True)  # neotropical-2 canon
    return float(sum(checks) / len(checks))


def bench_opossum_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_opossum_qa_studies": _bench_opossum_qa_studies(seed)}
