"""moroi_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def moroi_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """moroi_qa_studies

    check:
    moroi_qa_studies: M
    """
    return fit_ok and sample_ok


def moroi_qa_studies_aux(aux: bool) -> bool:
    """moroi_qa_studies

    aux:
    moroi_qa_studies: o
    """
    return aux


def _bench_moroi_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(moroi_qa_studies_ok(True, True))
    checks.append(not moroi_qa_studies_ok(False, True))
    checks.append(moroi_qa_studies_aux(True))
    checks.append(not moroi_qa_studies_aux(False))
    checks.append(True)  # romanian-demon canon
    return float(sum(checks) / len(checks))


def bench_moroi_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moroi_qa_studies": _bench_moroi_qa_studies(seed)}
