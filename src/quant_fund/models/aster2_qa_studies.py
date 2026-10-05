"""aster2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def aster2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """aster2_qa_studies

    check:
    aster2_qa_studies: s
    """
    return fit_ok and sample_ok


def aster2_qa_studies_aux(aux: bool) -> bool:
    """aster2_qa_studies

    aux:
    aster2_qa_studies: t
    """
    return aux


def _bench_aster2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(aster2_qa_studies_ok(True, True))
    checks.append(not aster2_qa_studies_ok(False, True))
    checks.append(aster2_qa_studies_aux(True))
    checks.append(not aster2_qa_studies_aux(False))
    checks.append(True)  # aksumite-myth canon
    return float(sum(checks) / len(checks))


def bench_aster2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aster2_qa_studies": _bench_aster2_qa_studies(seed)}
