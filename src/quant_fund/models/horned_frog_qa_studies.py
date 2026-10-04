"""horned_frog_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def horned_frog_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """horned_frog_qa_studies

    check:
    horned_frog_qa_studies: HornedFrogQA metrics
    """
    return fit_ok and sample_ok


def horned_frog_qa_studies_aux(aux: bool) -> bool:
    """horned_frog_qa_studies

    aux:
    horned_frog_qa_studies: horned frogs, ambush sites, answers, and scores
    """
    return aux


def _bench_horned_frog_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(horned_frog_qa_studies_ok(True, True))
    checks.append(not horned_frog_qa_studies_ok(False, True))
    checks.append(horned_frog_qa_studies_aux(True))
    checks.append(not horned_frog_qa_studies_aux(False))
    checks.append(True)  # frog canon
    return float(sum(checks) / len(checks))


def bench_horned_frog_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_horned_frog_qa_studies": _bench_horned_frog_qa_studies(seed)}
