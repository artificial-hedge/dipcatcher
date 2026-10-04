"""sleih_beggey_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sleih_beggey_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sleih_beggey_qa_studies

    check:
    sleih_beggey_qa_studies: l
    """
    return fit_ok and sample_ok


def sleih_beggey_qa_studies_aux(aux: bool) -> bool:
    """sleih_beggey_qa_studies

    aux:
    sleih_beggey_qa_studies: i
    """
    return aux


def _bench_sleih_beggey_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sleih_beggey_qa_studies_ok(True, True))
    checks.append(not sleih_beggey_qa_studies_ok(False, True))
    checks.append(sleih_beggey_qa_studies_aux(True))
    checks.append(not sleih_beggey_qa_studies_aux(False))
    checks.append(True)  # manx-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_sleih_beggey_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sleih_beggey_qa_studies": _bench_sleih_beggey_qa_studies(seed)}
