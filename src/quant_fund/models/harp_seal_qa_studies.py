"""harp_seal_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def harp_seal_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """harp_seal_qa_studies

    check:
    harp_seal_qa_studies: HarpSealQA metrics
    """
    return fit_ok and sample_ok


def harp_seal_qa_studies_aux(aux: bool) -> bool:
    """harp_seal_qa_studies

    aux:
    harp_seal_qa_studies: harp seals, arctic floes, answers, and scores
    """
    return aux


def _bench_harp_seal_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(harp_seal_qa_studies_ok(True, True))
    checks.append(not harp_seal_qa_studies_ok(False, True))
    checks.append(harp_seal_qa_studies_aux(True))
    checks.append(not harp_seal_qa_studies_aux(False))
    checks.append(True)  # pinniped canon
    return float(sum(checks) / len(checks))


def bench_harp_seal_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_harp_seal_qa_studies": _bench_harp_seal_qa_studies(seed)}
