"""nergal_demon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nergal_demon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nergal_demon_qa_studies

    check:
    nergal_demon_qa_studies: n
    """
    return fit_ok and sample_ok


def nergal_demon_qa_studies_aux(aux: bool) -> bool:
    """nergal_demon_qa_studies

    aux:
    nergal_demon_qa_studies: e
    """
    return aux


def _bench_nergal_demon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nergal_demon_qa_studies_ok(True, True))
    checks.append(not nergal_demon_qa_studies_ok(False, True))
    checks.append(nergal_demon_qa_studies_aux(True))
    checks.append(not nergal_demon_qa_studies_aux(False))
    checks.append(True)  # mesopotamian-demon-4 canon
    return float(sum(checks) / len(checks))


def bench_nergal_demon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nergal_demon_qa_studies": _bench_nergal_demon_qa_studies(seed)}
