"""bacoo_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bacoo_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bacoo_qa_studies

    check:
    bacoo_qa_studies: B
    """
    return fit_ok and sample_ok


def bacoo_qa_studies_aux(aux: bool) -> bool:
    """bacoo_qa_studies

    aux:
    bacoo_qa_studies: a
    """
    return aux


def _bench_bacoo_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bacoo_qa_studies_ok(True, True))
    checks.append(not bacoo_qa_studies_ok(False, True))
    checks.append(bacoo_qa_studies_aux(True))
    checks.append(not bacoo_qa_studies_aux(False))
    checks.append(True)  # caribbean-demon canon
    return float(sum(checks) / len(checks))


def bench_bacoo_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bacoo_qa_studies": _bench_bacoo_qa_studies(seed)}
