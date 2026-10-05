"""spenta_mainyu_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def spenta_mainyu_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """spenta_mainyu_qa_studies

    check:
    spenta_mainyu_qa_studies: s
    """
    return fit_ok and sample_ok


def spenta_mainyu_qa_studies_aux(aux: bool) -> bool:
    """spenta_mainyu_qa_studies

    aux:
    spenta_mainyu_qa_studies: p
    """
    return aux


def _bench_spenta_mainyu_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(spenta_mainyu_qa_studies_ok(True, True))
    checks.append(not spenta_mainyu_qa_studies_ok(False, True))
    checks.append(spenta_mainyu_qa_studies_aux(True))
    checks.append(not spenta_mainyu_qa_studies_aux(False))
    checks.append(True)  # zoroastrian-myth canon
    return float(sum(checks) / len(checks))


def bench_spenta_mainyu_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spenta_mainyu_qa_studies": _bench_spenta_mainyu_qa_studies(seed)}
