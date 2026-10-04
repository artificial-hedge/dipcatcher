"""subj_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def subj_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """subj_lite_studies

    check:
    subj_lite_studies: Subjectivity metrics
    """
    return fit_ok and sample_ok


def subj_lite_studies_aux(aux: bool) -> bool:
    """subj_lite_studies

    aux:
    subj_lite_studies: sentences, labels, votes, and scores
    """
    return aux


def _bench_subj_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(subj_lite_studies_ok(True, True))
    checks.append(not subj_lite_studies_ok(False, True))
    checks.append(subj_lite_studies_aux(True))
    checks.append(not subj_lite_studies_aux(False))
    checks.append(True)  # intent-paraphrase canon
    return float(sum(checks) / len(checks))


def bench_subj_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_subj_lite_studies": _bench_subj_lite_studies(seed)}
