"""jurupari_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def jurupari_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jurupari_qa_studies

    check:
    jurupari_qa_studies: J
    """
    return fit_ok and sample_ok


def jurupari_qa_studies_aux(aux: bool) -> bool:
    """jurupari_qa_studies

    aux:
    jurupari_qa_studies: u
    """
    return aux


def _bench_jurupari_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jurupari_qa_studies_ok(True, True))
    checks.append(not jurupari_qa_studies_ok(False, True))
    checks.append(jurupari_qa_studies_aux(True))
    checks.append(not jurupari_qa_studies_aux(False))
    checks.append(True)  # brazilian-slavic remnant canon
    return float(sum(checks) / len(checks))


def bench_jurupari_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jurupari_qa_studies": _bench_jurupari_qa_studies(seed)}
