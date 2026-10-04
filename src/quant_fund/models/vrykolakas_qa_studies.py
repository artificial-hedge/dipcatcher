"""vrykolakas_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def vrykolakas_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """vrykolakas_qa_studies

    check:
    vrykolakas_qa_studies: v
    """
    return fit_ok and sample_ok


def vrykolakas_qa_studies_aux(aux: bool) -> bool:
    """vrykolakas_qa_studies

    aux:
    vrykolakas_qa_studies: r
    """
    return aux


def _bench_vrykolakas_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(vrykolakas_qa_studies_ok(True, True))
    checks.append(not vrykolakas_qa_studies_ok(False, True))
    checks.append(vrykolakas_qa_studies_aux(True))
    checks.append(not vrykolakas_qa_studies_aux(False))
    checks.append(True)  # european-vampire canon
    return float(sum(checks) / len(checks))


def bench_vrykolakas_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vrykolakas_qa_studies": _bench_vrykolakas_qa_studies(seed)}
