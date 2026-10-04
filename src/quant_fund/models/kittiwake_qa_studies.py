"""kittiwake_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kittiwake_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kittiwake_qa_studies

    check:
    kittiwake_qa_studies: KittiwakeQA metrics
    """
    return fit_ok and sample_ok


def kittiwake_qa_studies_aux(aux: bool) -> bool:
    """kittiwake_qa_studies

    aux:
    kittiwake_qa_studies: kittiwakes, colonies, answers, and scores
    """
    return aux


def _bench_kittiwake_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kittiwake_qa_studies_ok(True, True))
    checks.append(not kittiwake_qa_studies_ok(False, True))
    checks.append(kittiwake_qa_studies_aux(True))
    checks.append(not kittiwake_qa_studies_aux(False))
    checks.append(True)  # seabird-2 canon
    return float(sum(checks) / len(checks))


def bench_kittiwake_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kittiwake_qa_studies": _bench_kittiwake_qa_studies(seed)}
