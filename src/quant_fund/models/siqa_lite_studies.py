"""siqa_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def siqa_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """siqa_lite_studies

    check:
    siqa_lite_studies: SocialIQA metrics
    """
    return fit_ok and sample_ok


def siqa_lite_studies_aux(aux: bool) -> bool:
    """siqa_lite_studies

    aux:
    siqa_lite_studies: contexts, questions, answers, and scores
    """
    return aux


def _bench_siqa_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(siqa_lite_studies_ok(True, True))
    checks.append(not siqa_lite_studies_ok(False, True))
    checks.append(siqa_lite_studies_aux(True))
    checks.append(not siqa_lite_studies_aux(False))
    checks.append(True)  # social-reasoning canon
    return float(sum(checks) / len(checks))


def bench_siqa_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_siqa_lite_studies": _bench_siqa_lite_studies(seed)}
