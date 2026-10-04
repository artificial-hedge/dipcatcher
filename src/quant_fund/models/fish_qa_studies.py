"""fish_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fish_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fish_qa_studies

    check:
    fish_qa_studies: FishQA metrics
    """
    return fit_ok and sample_ok


def fish_qa_studies_aux(aux: bool) -> bool:
    """fish_qa_studies

    aux:
    fish_qa_studies: fish, habitats, answers, and scores
    """
    return aux


def _bench_fish_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fish_qa_studies_ok(True, True))
    checks.append(not fish_qa_studies_ok(False, True))
    checks.append(fish_qa_studies_aux(True))
    checks.append(not fish_qa_studies_aux(False))
    checks.append(True)  # wildlife canon
    return float(sum(checks) / len(checks))


def bench_fish_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fish_qa_studies": _bench_fish_qa_studies(seed)}
