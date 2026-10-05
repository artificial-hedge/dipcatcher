"""purson_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def purson_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """purson_qa_studies

    check:
    purson_qa_studies: P
    """
    return fit_ok and sample_ok


def purson_qa_studies_aux(aux: bool) -> bool:
    """purson_qa_studies

    aux:
    purson_qa_studies: u
    """
    return aux


def _bench_purson_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(purson_qa_studies_ok(True, True))
    checks.append(not purson_qa_studies_ok(False, True))
    checks.append(purson_qa_studies_aux(True))
    checks.append(not purson_qa_studies_aux(False))
    checks.append(True)  # goetic-hierarchy canon
    return float(sum(checks) / len(checks))


def bench_purson_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_purson_qa_studies": _bench_purson_qa_studies(seed)}
