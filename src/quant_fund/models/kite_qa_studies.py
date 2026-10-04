"""kite_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kite_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kite_qa_studies

    check:
    kite_qa_studies: KiteQA metrics
    """
    return fit_ok and sample_ok


def kite_qa_studies_aux(aux: bool) -> bool:
    """kite_qa_studies

    aux:
    kite_qa_studies: kites, thermals, answers, and scores
    """
    return aux


def _bench_kite_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kite_qa_studies_ok(True, True))
    checks.append(not kite_qa_studies_ok(False, True))
    checks.append(kite_qa_studies_aux(True))
    checks.append(not kite_qa_studies_aux(False))
    checks.append(True)  # raptor canon
    return float(sum(checks) / len(checks))


def bench_kite_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kite_qa_studies": _bench_kite_qa_studies(seed)}
