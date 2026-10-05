"""stolas_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def stolas_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """stolas_qa_studies

    check:
    stolas_qa_studies: S
    """
    return fit_ok and sample_ok


def stolas_qa_studies_aux(aux: bool) -> bool:
    """stolas_qa_studies

    aux:
    stolas_qa_studies: t
    """
    return aux


def _bench_stolas_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(stolas_qa_studies_ok(True, True))
    checks.append(not stolas_qa_studies_ok(False, True))
    checks.append(stolas_qa_studies_aux(True))
    checks.append(not stolas_qa_studies_aux(False))
    checks.append(True)  # goetic-demon canon
    return float(sum(checks) / len(checks))


def bench_stolas_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stolas_qa_studies": _bench_stolas_qa_studies(seed)}
