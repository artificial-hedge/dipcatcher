"""plane_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def plane_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """plane_qa_studies

    check:
    plane_qa_studies: PlaneQA metrics
    """
    return fit_ok and sample_ok


def plane_qa_studies_aux(aux: bool) -> bool:
    """plane_qa_studies

    aux:
    plane_qa_studies: planes, flights, answers, and scores
    """
    return aux


def _bench_plane_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(plane_qa_studies_ok(True, True))
    checks.append(not plane_qa_studies_ok(False, True))
    checks.append(plane_qa_studies_aux(True))
    checks.append(not plane_qa_studies_aux(False))
    checks.append(True)  # vehicle canon
    return float(sum(checks) / len(checks))


def bench_plane_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_plane_qa_studies": _bench_plane_qa_studies(seed)}
