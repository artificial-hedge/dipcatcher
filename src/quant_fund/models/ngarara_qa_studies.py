"""ngarara_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ngarara_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ngarara_qa_studies

    check:
    ngarara_qa_studies: N
    """
    return fit_ok and sample_ok


def ngarara_qa_studies_aux(aux: bool) -> bool:
    """ngarara_qa_studies

    aux:
    ngarara_qa_studies: g
    """
    return aux


def _bench_ngarara_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ngarara_qa_studies_ok(True, True))
    checks.append(not ngarara_qa_studies_ok(False, True))
    checks.append(ngarara_qa_studies_aux(True))
    checks.append(not ngarara_qa_studies_aux(False))
    checks.append(True)  # polynesian-demon canon
    return float(sum(checks) / len(checks))


def bench_ngarara_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ngarara_qa_studies": _bench_ngarara_qa_studies(seed)}
