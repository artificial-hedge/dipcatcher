"""tanit_lok_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tanit_lok_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tanit_lok_qa_studies

    check:
    tanit_lok_qa_studies: p
    """
    return fit_ok and sample_ok


def tanit_lok_qa_studies_aux(aux: bool) -> bool:
    """tanit_lok_qa_studies

    aux:
    tanit_lok_qa_studies: u
    """
    return aux


def _bench_tanit_lok_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tanit_lok_qa_studies_ok(True, True))
    checks.append(not tanit_lok_qa_studies_ok(False, True))
    checks.append(tanit_lok_qa_studies_aux(True))
    checks.append(not tanit_lok_qa_studies_aux(False))
    checks.append(True)  # tuareg canon
    return float(sum(checks) / len(checks))


def bench_tanit_lok_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tanit_lok_qa_studies": _bench_tanit_lok_qa_studies(seed)}
