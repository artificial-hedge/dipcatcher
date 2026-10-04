"""berstuk_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def berstuk_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """berstuk_qa_studies

    check:
    berstuk_qa_studies: B
    """
    return fit_ok and sample_ok


def berstuk_qa_studies_aux(aux: bool) -> bool:
    """berstuk_qa_studies

    aux:
    berstuk_qa_studies: e
    """
    return aux


def _bench_berstuk_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(berstuk_qa_studies_ok(True, True))
    checks.append(not berstuk_qa_studies_ok(False, True))
    checks.append(berstuk_qa_studies_aux(True))
    checks.append(not berstuk_qa_studies_aux(False))
    checks.append(True)  # slavic-demon-3 canon
    return float(sum(checks) / len(checks))


def bench_berstuk_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_berstuk_qa_studies": _bench_berstuk_qa_studies(seed)}
