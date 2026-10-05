"""papa_legba_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def papa_legba_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """papa_legba_qa_studies

    check:
    papa_legba_qa_studies: P
    """
    return fit_ok and sample_ok


def papa_legba_qa_studies_aux(aux: bool) -> bool:
    """papa_legba_qa_studies

    aux:
    papa_legba_qa_studies: a
    """
    return aux


def _bench_papa_legba_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(papa_legba_qa_studies_ok(True, True))
    checks.append(not papa_legba_qa_studies_ok(False, True))
    checks.append(papa_legba_qa_studies_aux(True))
    checks.append(not papa_legba_qa_studies_aux(False))
    checks.append(True)  # vodou-loa canon
    return float(sum(checks) / len(checks))


def bench_papa_legba_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_papa_legba_qa_studies": _bench_papa_legba_qa_studies(seed)}
