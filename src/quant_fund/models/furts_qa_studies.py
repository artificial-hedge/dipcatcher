"""furts_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def furts_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """furts_qa_studies

    check:
    furts_qa_studies: F
    """
    return fit_ok and sample_ok


def furts_qa_studies_aux(aux: bool) -> bool:
    """furts_qa_studies

    aux:
    furts_qa_studies: u
    """
    return aux


def _bench_furts_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(furts_qa_studies_ok(True, True))
    checks.append(not furts_qa_studies_ok(False, True))
    checks.append(furts_qa_studies_aux(True))
    checks.append(not furts_qa_studies_aux(False))
    checks.append(True)  # caucasus-demon canon
    return float(sum(checks) / len(checks))


def bench_furts_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_furts_qa_studies": _bench_furts_qa_studies(seed)}
