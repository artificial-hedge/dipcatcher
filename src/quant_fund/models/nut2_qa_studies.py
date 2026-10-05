"""nut2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nut2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nut2_qa_studies

    check:
    nut2_qa_studies: Nut2QA metrics
    """
    return fit_ok and sample_ok


def nut2_qa_studies_aux(aux: bool) -> bool:
    """nut2_qa_studies

    aux:
    nut2_qa_studies: nut2, star vaults, answers, and scores
    """
    return aux


def _bench_nut2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nut2_qa_studies_ok(True, True))
    checks.append(not nut2_qa_studies_ok(False, True))
    checks.append(nut2_qa_studies_aux(True))
    checks.append(not nut2_qa_studies_aux(False))
    checks.append(True)  # egyptian-9 canon
    return float(sum(checks) / len(checks))


def bench_nut2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nut2_qa_studies": _bench_nut2_qa_studies(seed)}
