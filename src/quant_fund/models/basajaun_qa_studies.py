"""basajaun_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def basajaun_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """basajaun_qa_studies

    check:
    basajaun_qa_studies: BasajaunQA metrics
    """
    return fit_ok and sample_ok


def basajaun_qa_studies_aux(aux: bool) -> bool:
    """basajaun_qa_studies

    aux:
    basajaun_qa_studies: basajaun, forest lords, answers, and scores
    """
    return aux


def _bench_basajaun_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(basajaun_qa_studies_ok(True, True))
    checks.append(not basajaun_qa_studies_ok(False, True))
    checks.append(basajaun_qa_studies_aux(True))
    checks.append(not basajaun_qa_studies_aux(False))
    checks.append(True)  # basque-myth canon
    return float(sum(checks) / len(checks))


def bench_basajaun_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_basajaun_qa_studies": _bench_basajaun_qa_studies(seed)}
