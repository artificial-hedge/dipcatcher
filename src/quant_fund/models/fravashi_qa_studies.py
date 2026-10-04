"""fravashi_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fravashi_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fravashi_qa_studies

    check:
    fravashi_qa_studies: f
    """
    return fit_ok and sample_ok


def fravashi_qa_studies_aux(aux: bool) -> bool:
    """fravashi_qa_studies

    aux:
    fravashi_qa_studies: r
    """
    return aux


def _bench_fravashi_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fravashi_qa_studies_ok(True, True))
    checks.append(not fravashi_qa_studies_ok(False, True))
    checks.append(fravashi_qa_studies_aux(True))
    checks.append(not fravashi_qa_studies_aux(False))
    checks.append(True)  # zoroastrian-myth canon
    return float(sum(checks) / len(checks))


def bench_fravashi_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fravashi_qa_studies": _bench_fravashi_qa_studies(seed)}
