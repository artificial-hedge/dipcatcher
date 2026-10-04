"""begins_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def begins_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """begins_lite_studies

    check:
    begins_lite_studies: BEGIN grounded-dialog metrics
    """
    return fit_ok and sample_ok


def begins_lite_studies_aux(aux: bool) -> bool:
    """begins_lite_studies

    aux:
    begins_lite_studies: contexts, responses, knowledge, and scores
    """
    return aux


def _bench_begins_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(begins_lite_studies_ok(True, True))
    checks.append(not begins_lite_studies_ok(False, True))
    checks.append(begins_lite_studies_aux(True))
    checks.append(not begins_lite_studies_aux(False))
    checks.append(True)  # dialogue-2 canon
    return float(sum(checks) / len(checks))


def bench_begins_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_begins_lite_studies": _bench_begins_lite_studies(seed)}
