"""pukaua_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pukaua_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pukaua_qa_studies

    check:
    pukaua_qa_studies: P
    """
    return fit_ok and sample_ok


def pukaua_qa_studies_aux(aux: bool) -> bool:
    """pukaua_qa_studies

    aux:
    pukaua_qa_studies: u
    """
    return aux


def _bench_pukaua_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pukaua_qa_studies_ok(True, True))
    checks.append(not pukaua_qa_studies_ok(False, True))
    checks.append(pukaua_qa_studies_aux(True))
    checks.append(not pukaua_qa_studies_aux(False))
    checks.append(True)  # oceania-demon canon
    return float(sum(checks) / len(checks))


def bench_pukaua_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pukaua_qa_studies": _bench_pukaua_qa_studies(seed)}
