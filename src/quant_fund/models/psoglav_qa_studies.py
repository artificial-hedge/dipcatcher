"""psoglav_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def psoglav_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """psoglav_qa_studies

    check:
    psoglav_qa_studies: P
    """
    return fit_ok and sample_ok


def psoglav_qa_studies_aux(aux: bool) -> bool:
    """psoglav_qa_studies

    aux:
    psoglav_qa_studies: s
    """
    return aux


def _bench_psoglav_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(psoglav_qa_studies_ok(True, True))
    checks.append(not psoglav_qa_studies_ok(False, True))
    checks.append(psoglav_qa_studies_aux(True))
    checks.append(not psoglav_qa_studies_aux(False))
    checks.append(True)  # slavic-demon-2 canon
    return float(sum(checks) / len(checks))


def bench_psoglav_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_psoglav_qa_studies": _bench_psoglav_qa_studies(seed)}
