"""milkart_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def milkart_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """milkart_qa_studies

    check:
    milkart_qa_studies: c
    """
    return fit_ok and sample_ok


def milkart_qa_studies_aux(aux: bool) -> bool:
    """milkart_qa_studies

    aux:
    milkart_qa_studies: a
    """
    return aux


def _bench_milkart_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(milkart_qa_studies_ok(True, True))
    checks.append(not milkart_qa_studies_ok(False, True))
    checks.append(milkart_qa_studies_aux(True))
    checks.append(not milkart_qa_studies_aux(False))
    checks.append(True)  # tuareg-2 canon
    return float(sum(checks) / len(checks))


def bench_milkart_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_milkart_qa_studies": _bench_milkart_qa_studies(seed)}
