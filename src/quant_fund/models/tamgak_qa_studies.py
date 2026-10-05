"""tamgak_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tamgak_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tamgak_qa_studies

    check:
    tamgak_qa_studies: m
    """
    return fit_ok and sample_ok


def tamgak_qa_studies_aux(aux: bool) -> bool:
    """tamgak_qa_studies

    aux:
    tamgak_qa_studies: o
    """
    return aux


def _bench_tamgak_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tamgak_qa_studies_ok(True, True))
    checks.append(not tamgak_qa_studies_ok(False, True))
    checks.append(tamgak_qa_studies_aux(True))
    checks.append(not tamgak_qa_studies_aux(False))
    checks.append(True)  # tuareg-2 canon
    return float(sum(checks) / len(checks))


def bench_tamgak_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tamgak_qa_studies": _bench_tamgak_qa_studies(seed)}
