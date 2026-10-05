"""anbay_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def anbay_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """anbay_qa_studies

    check:
    anbay_qa_studies: w
    """
    return fit_ok and sample_ok


def anbay_qa_studies_aux(aux: bool) -> bool:
    """anbay_qa_studies

    aux:
    anbay_qa_studies: i
    """
    return aux


def _bench_anbay_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(anbay_qa_studies_ok(True, True))
    checks.append(not anbay_qa_studies_ok(False, True))
    checks.append(anbay_qa_studies_aux(True))
    checks.append(not anbay_qa_studies_aux(False))
    checks.append(True)  # sabaean-myth canon
    return float(sum(checks) / len(checks))


def bench_anbay_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_anbay_qa_studies": _bench_anbay_qa_studies(seed)}
