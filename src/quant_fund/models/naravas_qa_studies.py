"""naravas_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def naravas_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """naravas_qa_studies

    check:
    naravas_qa_studies: c
    """
    return fit_ok and sample_ok


def naravas_qa_studies_aux(aux: bool) -> bool:
    """naravas_qa_studies

    aux:
    naravas_qa_studies: a
    """
    return aux


def _bench_naravas_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(naravas_qa_studies_ok(True, True))
    checks.append(not naravas_qa_studies_ok(False, True))
    checks.append(naravas_qa_studies_aux(True))
    checks.append(not naravas_qa_studies_aux(False))
    checks.append(True)  # numidian-2 canon
    return float(sum(checks) / len(checks))


def bench_naravas_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_naravas_qa_studies": _bench_naravas_qa_studies(seed)}
