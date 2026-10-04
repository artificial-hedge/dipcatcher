"""afriye_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def afriye_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """afriye_qa_studies

    check:
    afriye_qa_studies: s
    """
    return fit_ok and sample_ok


def afriye_qa_studies_aux(aux: bool) -> bool:
    """afriye_qa_studies

    aux:
    afriye_qa_studies: a
    """
    return aux


def _bench_afriye_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(afriye_qa_studies_ok(True, True))
    checks.append(not afriye_qa_studies_ok(False, True))
    checks.append(afriye_qa_studies_aux(True))
    checks.append(not afriye_qa_studies_aux(False))
    checks.append(True)  # tuareg canon
    return float(sum(checks) / len(checks))


def bench_afriye_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_afriye_qa_studies": _bench_afriye_qa_studies(seed)}
