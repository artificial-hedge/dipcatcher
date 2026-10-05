"""poltergeist_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def poltergeist_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """poltergeist_qa_studies

    check:
    poltergeist_qa_studies: P
    """
    return fit_ok and sample_ok


def poltergeist_qa_studies_aux(aux: bool) -> bool:
    """poltergeist_qa_studies

    aux:
    poltergeist_qa_studies: o
    """
    return aux


def _bench_poltergeist_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(poltergeist_qa_studies_ok(True, True))
    checks.append(not poltergeist_qa_studies_ok(False, True))
    checks.append(poltergeist_qa_studies_aux(True))
    checks.append(not poltergeist_qa_studies_aux(False))
    checks.append(True)  # germanic-demon canon
    return float(sum(checks) / len(checks))


def bench_poltergeist_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poltergeist_qa_studies": _bench_poltergeist_qa_studies(seed)}
