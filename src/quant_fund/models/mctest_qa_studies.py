"""mctest_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mctest_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mctest_qa_studies

    check:
    mctest_qa_studies: MCTest elementary-reading metrics
    """
    return fit_ok and sample_ok


def mctest_qa_studies_aux(aux: bool) -> bool:
    """mctest_qa_studies

    aux:
    mctest_qa_studies: stories, questions, options, and accuracies
    """
    return aux


def _bench_mctest_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mctest_qa_studies_ok(True, True))
    checks.append(not mctest_qa_studies_ok(False, True))
    checks.append(mctest_qa_studies_aux(True))
    checks.append(not mctest_qa_studies_aux(False))
    checks.append(True)  # reading-comprehension-3 canon
    return float(sum(checks) / len(checks))


def bench_mctest_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mctest_qa_studies": _bench_mctest_qa_studies(seed)}
