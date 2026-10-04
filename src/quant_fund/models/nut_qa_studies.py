"""nut_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nut_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nut_qa_studies

    check:
    nut_qa_studies: NutQA metrics
    """
    return fit_ok and sample_ok


def nut_qa_studies_aux(aux: bool) -> bool:
    """nut_qa_studies

    aux:
    nut_qa_studies: nut, star vaults, answers, and scores
    """
    return aux


def _bench_nut_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nut_qa_studies_ok(True, True))
    checks.append(not nut_qa_studies_ok(False, True))
    checks.append(nut_qa_studies_aux(True))
    checks.append(not nut_qa_studies_aux(False))
    checks.append(True)  # egyptian-4 canon
    return float(sum(checks) / len(checks))


def bench_nut_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nut_qa_studies": _bench_nut_qa_studies(seed)}
