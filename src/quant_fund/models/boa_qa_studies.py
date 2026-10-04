"""boa_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def boa_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """boa_qa_studies

    check:
    boa_qa_studies: BoaQA metrics
    """
    return fit_ok and sample_ok


def boa_qa_studies_aux(aux: bool) -> bool:
    """boa_qa_studies

    aux:
    boa_qa_studies: boas, constrictions, answers, and scores
    """
    return aux


def _bench_boa_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(boa_qa_studies_ok(True, True))
    checks.append(not boa_qa_studies_ok(False, True))
    checks.append(boa_qa_studies_aux(True))
    checks.append(not boa_qa_studies_aux(False))
    checks.append(True)  # reptile canon
    return float(sum(checks) / len(checks))


def bench_boa_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_boa_qa_studies": _bench_boa_qa_studies(seed)}
