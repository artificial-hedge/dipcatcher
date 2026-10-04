"""angelfish_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def angelfish_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """angelfish_qa_studies

    check:
    angelfish_qa_studies: AngelfishQA metrics
    """
    return fit_ok and sample_ok


def angelfish_qa_studies_aux(aux: bool) -> bool:
    """angelfish_qa_studies

    aux:
    angelfish_qa_studies: angelfish, reef crevices, answers, and scores
    """
    return aux


def _bench_angelfish_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(angelfish_qa_studies_ok(True, True))
    checks.append(not angelfish_qa_studies_ok(False, True))
    checks.append(angelfish_qa_studies_aux(True))
    checks.append(not angelfish_qa_studies_aux(False))
    checks.append(True)  # reef-fish-2 canon
    return float(sum(checks) / len(checks))


def bench_angelfish_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_angelfish_qa_studies": _bench_angelfish_qa_studies(seed)}
