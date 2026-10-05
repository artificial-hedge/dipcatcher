"""el2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def el2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """el2_qa_studies

    check:
    el2_qa_studies: m
    """
    return fit_ok and sample_ok


def el2_qa_studies_aux(aux: bool) -> bool:
    """el2_qa_studies

    aux:
    el2_qa_studies: o
    """
    return aux


def _bench_el2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(el2_qa_studies_ok(True, True))
    checks.append(not el2_qa_studies_ok(False, True))
    checks.append(el2_qa_studies_aux(True))
    checks.append(not el2_qa_studies_aux(False))
    checks.append(True)  # ammonite-myth canon
    return float(sum(checks) / len(checks))


def bench_el2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_el2_qa_studies": _bench_el2_qa_studies(seed)}
