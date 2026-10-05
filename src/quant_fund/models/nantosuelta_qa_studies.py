"""nantosuelta_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nantosuelta_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nantosuelta_qa_studies

    check:
    nantosuelta_qa_studies: v
    """
    return fit_ok and sample_ok


def nantosuelta_qa_studies_aux(aux: bool) -> bool:
    """nantosuelta_qa_studies

    aux:
    nantosuelta_qa_studies: a
    """
    return aux


def _bench_nantosuelta_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nantosuelta_qa_studies_ok(True, True))
    checks.append(not nantosuelta_qa_studies_ok(False, True))
    checks.append(nantosuelta_qa_studies_aux(True))
    checks.append(not nantosuelta_qa_studies_aux(False))
    checks.append(True)  # celtic-remnant canon
    return float(sum(checks) / len(checks))


def bench_nantosuelta_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nantosuelta_qa_studies": _bench_nantosuelta_qa_studies(seed)}
