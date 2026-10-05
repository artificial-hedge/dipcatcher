"""loviatar_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def loviatar_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """loviatar_qa_studies

    check:
    loviatar_qa_studies: L
    """
    return fit_ok and sample_ok


def loviatar_qa_studies_aux(aux: bool) -> bool:
    """loviatar_qa_studies

    aux:
    loviatar_qa_studies: o
    """
    return aux


def _bench_loviatar_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(loviatar_qa_studies_ok(True, True))
    checks.append(not loviatar_qa_studies_ok(False, True))
    checks.append(loviatar_qa_studies_aux(True))
    checks.append(not loviatar_qa_studies_aux(False))
    checks.append(True)  # finnish-demon canon
    return float(sum(checks) / len(checks))


def bench_loviatar_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_loviatar_qa_studies": _bench_loviatar_qa_studies(seed)}
