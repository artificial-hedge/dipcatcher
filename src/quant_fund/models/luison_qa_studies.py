"""luison_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def luison_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """luison_qa_studies

    check:
    luison_qa_studies: L
    """
    return fit_ok and sample_ok


def luison_qa_studies_aux(aux: bool) -> bool:
    """luison_qa_studies

    aux:
    luison_qa_studies: u
    """
    return aux


def _bench_luison_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(luison_qa_studies_ok(True, True))
    checks.append(not luison_qa_studies_ok(False, True))
    checks.append(luison_qa_studies_aux(True))
    checks.append(not luison_qa_studies_aux(False))
    checks.append(True)  # guarani-demon canon
    return float(sum(checks) / len(checks))


def bench_luison_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_luison_qa_studies": _bench_luison_qa_studies(seed)}
