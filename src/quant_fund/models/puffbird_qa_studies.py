"""puffbird_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def puffbird_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """puffbird_qa_studies

    check:
    puffbird_qa_studies: PuffbirdQA metrics
    """
    return fit_ok and sample_ok


def puffbird_qa_studies_aux(aux: bool) -> bool:
    """puffbird_qa_studies

    aux:
    puffbird_qa_studies: puffbirds, understories, answers, and scores
    """
    return aux


def _bench_puffbird_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(puffbird_qa_studies_ok(True, True))
    checks.append(not puffbird_qa_studies_ok(False, True))
    checks.append(puffbird_qa_studies_aux(True))
    checks.append(not puffbird_qa_studies_aux(False))
    checks.append(True)  # coraciiform canon
    return float(sum(checks) / len(checks))


def bench_puffbird_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_puffbird_qa_studies": _bench_puffbird_qa_studies(seed)}
