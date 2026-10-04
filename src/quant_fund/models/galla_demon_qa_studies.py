"""galla_demon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def galla_demon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """galla_demon_qa_studies

    check:
    galla_demon_qa_studies: g
    """
    return fit_ok and sample_ok


def galla_demon_qa_studies_aux(aux: bool) -> bool:
    """galla_demon_qa_studies

    aux:
    galla_demon_qa_studies: a
    """
    return aux


def _bench_galla_demon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(galla_demon_qa_studies_ok(True, True))
    checks.append(not galla_demon_qa_studies_ok(False, True))
    checks.append(galla_demon_qa_studies_aux(True))
    checks.append(not galla_demon_qa_studies_aux(False))
    checks.append(True)  # mesopotamian-demon canon
    return float(sum(checks) / len(checks))


def bench_galla_demon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_galla_demon_qa_studies": _bench_galla_demon_qa_studies(seed)}
