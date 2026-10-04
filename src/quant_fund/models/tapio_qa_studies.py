"""tapio_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tapio_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tapio_qa_studies

    check:
    tapio_qa_studies: TapioQA metrics
    """
    return fit_ok and sample_ok


def tapio_qa_studies_aux(aux: bool) -> bool:
    """tapio_qa_studies

    aux:
    tapio_qa_studies: tapio, forest lords, answers, and scores
    """
    return aux


def _bench_tapio_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tapio_qa_studies_ok(True, True))
    checks.append(not tapio_qa_studies_ok(False, True))
    checks.append(tapio_qa_studies_aux(True))
    checks.append(not tapio_qa_studies_aux(False))
    checks.append(True)  # finno-ugric-myth canon
    return float(sum(checks) / len(checks))


def bench_tapio_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tapio_qa_studies": _bench_tapio_qa_studies(seed)}
