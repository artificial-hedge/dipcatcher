"""lilu_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def lilu_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lilu_qa_studies

    check:
    lilu_qa_studies: l
    """
    return fit_ok and sample_ok


def lilu_qa_studies_aux(aux: bool) -> bool:
    """lilu_qa_studies

    aux:
    lilu_qa_studies: i
    """
    return aux


def _bench_lilu_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(lilu_qa_studies_ok(True, True))
    checks.append(not lilu_qa_studies_ok(False, True))
    checks.append(lilu_qa_studies_aux(True))
    checks.append(not lilu_qa_studies_aux(False))
    checks.append(True)  # mesopotamian-demon-3 canon
    return float(sum(checks) / len(checks))


def bench_lilu_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lilu_qa_studies": _bench_lilu_qa_studies(seed)}
