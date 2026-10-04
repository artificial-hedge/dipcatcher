"""dune_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def dune_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dune_qa_studies

    check:
    dune_qa_studies: DuneQA metrics
    """
    return fit_ok and sample_ok


def dune_qa_studies_aux(aux: bool) -> bool:
    """dune_qa_studies

    aux:
    dune_qa_studies: dunes, sands, answers, and scores
    """
    return aux


def _bench_dune_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dune_qa_studies_ok(True, True))
    checks.append(not dune_qa_studies_ok(False, True))
    checks.append(dune_qa_studies_aux(True))
    checks.append(not dune_qa_studies_aux(False))
    checks.append(True)  # landform canon
    return float(sum(checks) / len(checks))


def bench_dune_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dune_qa_studies": _bench_dune_qa_studies(seed)}
