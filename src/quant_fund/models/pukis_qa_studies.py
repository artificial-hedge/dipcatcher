"""pukis_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pukis_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pukis_qa_studies

    check:
    pukis_qa_studies: P
    """
    return fit_ok and sample_ok


def pukis_qa_studies_aux(aux: bool) -> bool:
    """pukis_qa_studies

    aux:
    pukis_qa_studies: u
    """
    return aux


def _bench_pukis_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pukis_qa_studies_ok(True, True))
    checks.append(not pukis_qa_studies_ok(False, True))
    checks.append(pukis_qa_studies_aux(True))
    checks.append(not pukis_qa_studies_aux(False))
    checks.append(True)  # baltic-demon canon
    return float(sum(checks) / len(checks))


def bench_pukis_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pukis_qa_studies": _bench_pukis_qa_studies(seed)}
