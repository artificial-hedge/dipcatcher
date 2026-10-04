"""macaw_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def macaw_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """macaw_qa_studies

    check:
    macaw_qa_studies: MacawQA metrics
    """
    return fit_ok and sample_ok


def macaw_qa_studies_aux(aux: bool) -> bool:
    """macaw_qa_studies

    aux:
    macaw_qa_studies: macaws, feathers, answers, and scores
    """
    return aux


def _bench_macaw_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(macaw_qa_studies_ok(True, True))
    checks.append(not macaw_qa_studies_ok(False, True))
    checks.append(macaw_qa_studies_aux(True))
    checks.append(not macaw_qa_studies_aux(False))
    checks.append(True)  # jungle canon
    return float(sum(checks) / len(checks))


def bench_macaw_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_macaw_qa_studies": _bench_macaw_qa_studies(seed)}
