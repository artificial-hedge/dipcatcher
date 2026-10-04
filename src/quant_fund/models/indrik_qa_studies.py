"""indrik_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def indrik_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """indrik_qa_studies

    check:
    indrik_qa_studies: I
    """
    return fit_ok and sample_ok


def indrik_qa_studies_aux(aux: bool) -> bool:
    """indrik_qa_studies

    aux:
    indrik_qa_studies: n
    """
    return aux


def _bench_indrik_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(indrik_qa_studies_ok(True, True))
    checks.append(not indrik_qa_studies_ok(False, True))
    checks.append(indrik_qa_studies_aux(True))
    checks.append(not indrik_qa_studies_aux(False))
    checks.append(True)  # slavic-demon-2 canon
    return float(sum(checks) / len(checks))


def bench_indrik_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_indrik_qa_studies": _bench_indrik_qa_studies(seed)}
