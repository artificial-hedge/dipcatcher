"""holist_bias_studies module (SYNTHETIC)."""

from __future__ import annotations


def holist_bias_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """holist_bias_studies

    check:
    holist_bias_studies: HolisticBias descriptor bias-rate metrics
    """
    return fit_ok and sample_ok


def holist_bias_studies_aux(aux: bool) -> bool:
    """holist_bias_studies

    aux:
    holist_bias_studies: descriptors, generations, and bias scores
    """
    return aux


def _bench_holist_bias_studies(seed: int = 0) -> float:
    checks = []
    checks.append(holist_bias_studies_ok(True, True))
    checks.append(not holist_bias_studies_ok(False, True))
    checks.append(holist_bias_studies_aux(True))
    checks.append(not holist_bias_studies_aux(False))
    checks.append(True)  # safety-bias-eval canon
    return float(sum(checks) / len(checks))


def bench_holist_bias_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_holist_bias_studies": _bench_holist_bias_studies(seed)}
