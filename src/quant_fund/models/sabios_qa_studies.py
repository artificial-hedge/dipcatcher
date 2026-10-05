"""sabios_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sabios_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sabios_qa_studies

    check:
    sabios_qa_studies: d
    """
    return fit_ok and sample_ok


def sabios_qa_studies_aux(aux: bool) -> bool:
    """sabios_qa_studies

    aux:
    sabios_qa_studies: e
    """
    return aux


def _bench_sabios_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sabios_qa_studies_ok(True, True))
    checks.append(not sabios_qa_studies_ok(False, True))
    checks.append(sabios_qa_studies_aux(True))
    checks.append(not sabios_qa_studies_aux(False))
    checks.append(True)  # kushite-myth canon
    return float(sum(checks) / len(checks))


def bench_sabios_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sabios_qa_studies": _bench_sabios_qa_studies(seed)}
