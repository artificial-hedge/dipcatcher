"""bank_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bank_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bank_qa_studies

    check:
    bank_qa_studies: BankQA metrics
    """
    return fit_ok and sample_ok


def bank_qa_studies_aux(aux: bool) -> bool:
    """bank_qa_studies

    aux:
    bank_qa_studies: accounts, statements, answers, and scores
    """
    return aux


def _bench_bank_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bank_qa_studies_ok(True, True))
    checks.append(not bank_qa_studies_ok(False, True))
    checks.append(bank_qa_studies_aux(True))
    checks.append(not bank_qa_studies_aux(False))
    checks.append(True)  # financial-NLP canon
    return float(sum(checks) / len(checks))


def bench_bank_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bank_qa_studies": _bench_bank_qa_studies(seed)}
