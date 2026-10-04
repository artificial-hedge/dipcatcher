"""atlas_moth_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def atlas_moth_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """atlas_moth_qa_studies

    check:
    atlas_moth_qa_studies: AtlasMothQA metrics
    """
    return fit_ok and sample_ok


def atlas_moth_qa_studies_aux(aux: bool) -> bool:
    """atlas_moth_qa_studies

    aux:
    atlas_moth_qa_studies: atlas moths, jungles, answers, and scores
    """
    return aux


def _bench_atlas_moth_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(atlas_moth_qa_studies_ok(True, True))
    checks.append(not atlas_moth_qa_studies_ok(False, True))
    checks.append(atlas_moth_qa_studies_aux(True))
    checks.append(not atlas_moth_qa_studies_aux(False))
    checks.append(True)  # moth canon
    return float(sum(checks) / len(checks))


def bench_atlas_moth_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_atlas_moth_qa_studies": _bench_atlas_moth_qa_studies(seed)}
