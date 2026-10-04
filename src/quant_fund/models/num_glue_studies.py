"""num_glue_studies module (SYNTHETIC)."""

from __future__ import annotations


def num_glue_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """num_glue_studies

    check:
    num_glue_studies: NumGLUE metrics
    """
    return fit_ok and sample_ok


def num_glue_studies_aux(aux: bool) -> bool:
    """num_glue_studies

    aux:
    num_glue_studies: questions, formats, answers, and scores
    """
    return aux


def _bench_num_glue_studies(seed: int = 0) -> float:
    checks = []
    checks.append(num_glue_studies_ok(True, True))
    checks.append(not num_glue_studies_ok(False, True))
    checks.append(num_glue_studies_aux(True))
    checks.append(not num_glue_studies_aux(False))
    checks.append(True)  # numerical-reasoning canon
    return float(sum(checks) / len(checks))


def bench_num_glue_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_num_glue_studies": _bench_num_glue_studies(seed)}
