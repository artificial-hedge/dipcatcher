"""mahr_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mahr_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mahr_qa_studies

    check:
    mahr_qa_studies: M
    """
    return fit_ok and sample_ok


def mahr_qa_studies_aux(aux: bool) -> bool:
    """mahr_qa_studies

    aux:
    mahr_qa_studies: a
    """
    return aux


def _bench_mahr_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mahr_qa_studies_ok(True, True))
    checks.append(not mahr_qa_studies_ok(False, True))
    checks.append(mahr_qa_studies_aux(True))
    checks.append(not mahr_qa_studies_aux(False))
    checks.append(True)  # germanic-demon canon
    return float(sum(checks) / len(checks))


def bench_mahr_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mahr_qa_studies": _bench_mahr_qa_studies(seed)}
