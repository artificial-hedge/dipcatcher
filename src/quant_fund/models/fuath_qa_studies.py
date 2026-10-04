"""fuath_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fuath_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fuath_qa_studies

    check:
    fuath_qa_studies: F
    """
    return fit_ok and sample_ok


def fuath_qa_studies_aux(aux: bool) -> bool:
    """fuath_qa_studies

    aux:
    fuath_qa_studies: u
    """
    return aux


def _bench_fuath_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fuath_qa_studies_ok(True, True))
    checks.append(not fuath_qa_studies_ok(False, True))
    checks.append(fuath_qa_studies_aux(True))
    checks.append(not fuath_qa_studies_aux(False))
    checks.append(True)  # celtic-demon-2 canon
    return float(sum(checks) / len(checks))


def bench_fuath_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fuath_qa_studies": _bench_fuath_qa_studies(seed)}
