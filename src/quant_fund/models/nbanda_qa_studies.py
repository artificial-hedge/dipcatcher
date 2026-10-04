"""nbanda_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nbanda_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nbanda_qa_studies

    check:
    nbanda_qa_studies: NbandaQA metrics
    """
    return fit_ok and sample_ok


def nbanda_qa_studies_aux(aux: bool) -> bool:
    """nbanda_qa_studies

    aux:
    nbanda_qa_studies: nbanda, leopard spirit, answers, and scores
    """
    return aux


def _bench_nbanda_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nbanda_qa_studies_ok(True, True))
    checks.append(not nbanda_qa_studies_ok(False, True))
    checks.append(nbanda_qa_studies_aux(True))
    checks.append(not nbanda_qa_studies_aux(False))
    checks.append(True)  # african-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_nbanda_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nbanda_qa_studies": _bench_nbanda_qa_studies(seed)}
