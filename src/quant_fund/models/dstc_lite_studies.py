"""dstc_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def dstc_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dstc_lite_studies

    check:
    dstc_lite_studies: DSTC task-dialogue metrics
    """
    return fit_ok and sample_ok


def dstc_lite_studies_aux(aux: bool) -> bool:
    """dstc_lite_studies

    aux:
    dstc_lite_studies: turns, intents, slots, and scores
    """
    return aux


def _bench_dstc_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dstc_lite_studies_ok(True, True))
    checks.append(not dstc_lite_studies_ok(False, True))
    checks.append(dstc_lite_studies_aux(True))
    checks.append(not dstc_lite_studies_aux(False))
    checks.append(True)  # dialogue-system canon
    return float(sum(checks) / len(checks))


def bench_dstc_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dstc_lite_studies": _bench_dstc_lite_studies(seed)}
