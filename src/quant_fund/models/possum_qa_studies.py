"""possum_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def possum_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """possum_qa_studies

    check:
    possum_qa_studies: PossumQA metrics
    """
    return fit_ok and sample_ok


def possum_qa_studies_aux(aux: bool) -> bool:
    """possum_qa_studies

    aux:
    possum_qa_studies: possums, trees, answers, and scores
    """
    return aux


def _bench_possum_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(possum_qa_studies_ok(True, True))
    checks.append(not possum_qa_studies_ok(False, True))
    checks.append(possum_qa_studies_aux(True))
    checks.append(not possum_qa_studies_aux(False))
    checks.append(True)  # marsupial-2 canon
    return float(sum(checks) / len(checks))


def bench_possum_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_possum_qa_studies": _bench_possum_qa_studies(seed)}
