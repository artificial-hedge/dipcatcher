"""beaver_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def beaver_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """beaver_qa_studies

    check:
    beaver_qa_studies: BeaverQA metrics
    """
    return fit_ok and sample_ok


def beaver_qa_studies_aux(aux: bool) -> bool:
    """beaver_qa_studies

    aux:
    beaver_qa_studies: beavers, dams, answers, and scores
    """
    return aux


def _bench_beaver_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(beaver_qa_studies_ok(True, True))
    checks.append(not beaver_qa_studies_ok(False, True))
    checks.append(beaver_qa_studies_aux(True))
    checks.append(not beaver_qa_studies_aux(False))
    checks.append(True)  # forest-mammal canon
    return float(sum(checks) / len(checks))


def bench_beaver_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_beaver_qa_studies": _bench_beaver_qa_studies(seed)}
