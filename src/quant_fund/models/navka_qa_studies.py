"""navka_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def navka_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """navka_qa_studies

    check:
    navka_qa_studies: N
    """
    return fit_ok and sample_ok


def navka_qa_studies_aux(aux: bool) -> bool:
    """navka_qa_studies

    aux:
    navka_qa_studies: a
    """
    return aux


def _bench_navka_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(navka_qa_studies_ok(True, True))
    checks.append(not navka_qa_studies_ok(False, True))
    checks.append(navka_qa_studies_aux(True))
    checks.append(not navka_qa_studies_aux(False))
    checks.append(True)  # slavic-demon-3 canon
    return float(sum(checks) / len(checks))


def bench_navka_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_navka_qa_studies": _bench_navka_qa_studies(seed)}
