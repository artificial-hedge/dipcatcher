"""comet_mt_studies module (SYNTHETIC)."""

from __future__ import annotations


def comet_mt_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """comet_mt_studies

    check:
    comet_mt_studies: COMET MT-eval metrics
    """
    return fit_ok and sample_ok


def comet_mt_studies_aux(aux: bool) -> bool:
    """comet_mt_studies

    aux:
    comet_mt_studies: sources, hypotheses, references, and scores
    """
    return aux


def _bench_comet_mt_studies(seed: int = 0) -> float:
    checks = []
    checks.append(comet_mt_studies_ok(True, True))
    checks.append(not comet_mt_studies_ok(False, True))
    checks.append(comet_mt_studies_aux(True))
    checks.append(not comet_mt_studies_aux(False))
    checks.append(True)  # generation-metric canon
    return float(sum(checks) / len(checks))


def bench_comet_mt_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_comet_mt_studies": _bench_comet_mt_studies(seed)}
