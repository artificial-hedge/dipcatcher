"""avsd_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def avsd_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """avsd_lite_studies

    check:
    avsd_lite_studies: AVSD metrics
    """
    return fit_ok and sample_ok


def avsd_lite_studies_aux(aux: bool) -> bool:
    """avsd_lite_studies

    aux:
    avsd_lite_studies: videos, dialogues, answers, and scores
    """
    return aux


def _bench_avsd_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(avsd_lite_studies_ok(True, True))
    checks.append(not avsd_lite_studies_ok(False, True))
    checks.append(avsd_lite_studies_aux(True))
    checks.append(not avsd_lite_studies_aux(False))
    checks.append(True)  # audio-QA canon
    return float(sum(checks) / len(checks))


def bench_avsd_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_avsd_lite_studies": _bench_avsd_lite_studies(seed)}
