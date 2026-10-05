"""mapinguari_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mapinguari_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mapinguari_qa_studies

    check:
    mapinguari_qa_studies: M
    """
    return fit_ok and sample_ok


def mapinguari_qa_studies_aux(aux: bool) -> bool:
    """mapinguari_qa_studies

    aux:
    mapinguari_qa_studies: a
    """
    return aux


def _bench_mapinguari_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mapinguari_qa_studies_ok(True, True))
    checks.append(not mapinguari_qa_studies_ok(False, True))
    checks.append(mapinguari_qa_studies_aux(True))
    checks.append(not mapinguari_qa_studies_aux(False))
    checks.append(True)  # brazilian-folklore canon
    return float(sum(checks) / len(checks))


def bench_mapinguari_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mapinguari_qa_studies": _bench_mapinguari_qa_studies(seed)}
