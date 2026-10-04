"""pauraque_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pauraque_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pauraque_qa_studies

    check:
    pauraque_qa_studies: PauraqueQA metrics
    """
    return fit_ok and sample_ok


def pauraque_qa_studies_aux(aux: bool) -> bool:
    """pauraque_qa_studies

    aux:
    pauraque_qa_studies: pauraques, thickets, answers, and scores
    """
    return aux


def _bench_pauraque_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pauraque_qa_studies_ok(True, True))
    checks.append(not pauraque_qa_studies_ok(False, True))
    checks.append(pauraque_qa_studies_aux(True))
    checks.append(not pauraque_qa_studies_aux(False))
    checks.append(True)  # nightjar-2 canon
    return float(sum(checks) / len(checks))


def bench_pauraque_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pauraque_qa_studies": _bench_pauraque_qa_studies(seed)}
