"""abduction_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def abduction_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """abduction_lite_studies

    check:
    abduction_lite_studies: Abductive-reasoning metrics
    """
    return fit_ok and sample_ok


def abduction_lite_studies_aux(aux: bool) -> bool:
    """abduction_lite_studies

    aux:
    abduction_lite_studies: observations, hypotheses, choices, and scores
    """
    return aux


def _bench_abduction_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(abduction_lite_studies_ok(True, True))
    checks.append(not abduction_lite_studies_ok(False, True))
    checks.append(abduction_lite_studies_aux(True))
    checks.append(not abduction_lite_studies_aux(False))
    checks.append(True)  # NLU-exotics canon
    return float(sum(checks) / len(checks))


def bench_abduction_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_abduction_lite_studies": _bench_abduction_lite_studies(seed)}
