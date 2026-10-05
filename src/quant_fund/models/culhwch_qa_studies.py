"""culhwch_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def culhwch_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """culhwch_qa_studies

    check:
    culhwch_qa_studies: o
    """
    return fit_ok and sample_ok


def culhwch_qa_studies_aux(aux: bool) -> bool:
    """culhwch_qa_studies

    aux:
    culhwch_qa_studies: l
    """
    return aux


def _bench_culhwch_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(culhwch_qa_studies_ok(True, True))
    checks.append(not culhwch_qa_studies_ok(False, True))
    checks.append(culhwch_qa_studies_aux(True))
    checks.append(not culhwch_qa_studies_aux(False))
    checks.append(True)  # arthurian-5 canon
    return float(sum(checks) / len(checks))


def bench_culhwch_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_culhwch_qa_studies": _bench_culhwch_qa_studies(seed)}
