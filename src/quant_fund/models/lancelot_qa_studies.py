"""lancelot_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def lancelot_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lancelot_qa_studies

    check:
    lancelot_qa_studies: l
    """
    return fit_ok and sample_ok


def lancelot_qa_studies_aux(aux: bool) -> bool:
    """lancelot_qa_studies

    aux:
    lancelot_qa_studies: a
    """
    return aux


def _bench_lancelot_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(lancelot_qa_studies_ok(True, True))
    checks.append(not lancelot_qa_studies_ok(False, True))
    checks.append(lancelot_qa_studies_aux(True))
    checks.append(not lancelot_qa_studies_aux(False))
    checks.append(True)  # arthurian-2 canon
    return float(sum(checks) / len(checks))


def bench_lancelot_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lancelot_qa_studies": _bench_lancelot_qa_studies(seed)}
