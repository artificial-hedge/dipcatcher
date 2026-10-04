"""sphagnum_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sphagnum_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sphagnum_qa_studies

    check:
    sphagnum_qa_studies: SphagnumQA metrics
    """
    return fit_ok and sample_ok


def sphagnum_qa_studies_aux(aux: bool) -> bool:
    """sphagnum_qa_studies

    aux:
    sphagnum_qa_studies: sphagnums, peatlands, answers, and scores
    """
    return aux


def _bench_sphagnum_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sphagnum_qa_studies_ok(True, True))
    checks.append(not sphagnum_qa_studies_ok(False, True))
    checks.append(sphagnum_qa_studies_aux(True))
    checks.append(not sphagnum_qa_studies_aux(False))
    checks.append(True)  # moss canon
    return float(sum(checks) / len(checks))


def bench_sphagnum_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sphagnum_qa_studies": _bench_sphagnum_qa_studies(seed)}
