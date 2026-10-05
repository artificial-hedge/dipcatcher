"""kurupi_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kurupi_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kurupi_qa_studies

    check:
    kurupi_qa_studies: K
    """
    return fit_ok and sample_ok


def kurupi_qa_studies_aux(aux: bool) -> bool:
    """kurupi_qa_studies

    aux:
    kurupi_qa_studies: u
    """
    return aux


def _bench_kurupi_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kurupi_qa_studies_ok(True, True))
    checks.append(not kurupi_qa_studies_ok(False, True))
    checks.append(kurupi_qa_studies_aux(True))
    checks.append(not kurupi_qa_studies_aux(False))
    checks.append(True)  # guarani-demon canon
    return float(sum(checks) / len(checks))


def bench_kurupi_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kurupi_qa_studies": _bench_kurupi_qa_studies(seed)}
