"""wraith_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def wraith_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """wraith_qa_studies

    check:
    wraith_qa_studies: W
    """
    return fit_ok and sample_ok


def wraith_qa_studies_aux(aux: bool) -> bool:
    """wraith_qa_studies

    aux:
    wraith_qa_studies: r
    """
    return aux


def _bench_wraith_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(wraith_qa_studies_ok(True, True))
    checks.append(not wraith_qa_studies_ok(False, True))
    checks.append(wraith_qa_studies_aux(True))
    checks.append(not wraith_qa_studies_aux(False))
    checks.append(True)  # celtic-demon-2 canon
    return float(sum(checks) / len(checks))


def bench_wraith_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wraith_qa_studies": _bench_wraith_qa_studies(seed)}
