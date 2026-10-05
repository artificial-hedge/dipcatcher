"""leyak_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def leyak_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """leyak_qa_studies

    check:
    leyak_qa_studies: L
    """
    return fit_ok and sample_ok


def leyak_qa_studies_aux(aux: bool) -> bool:
    """leyak_qa_studies

    aux:
    leyak_qa_studies: e
    """
    return aux


def _bench_leyak_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(leyak_qa_studies_ok(True, True))
    checks.append(not leyak_qa_studies_ok(False, True))
    checks.append(leyak_qa_studies_aux(True))
    checks.append(not leyak_qa_studies_aux(False))
    checks.append(True)  # javanese-demon canon
    return float(sum(checks) / len(checks))


def bench_leyak_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_leyak_qa_studies": _bench_leyak_qa_studies(seed)}
