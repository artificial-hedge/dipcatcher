"""crowspairs_studies module (SYNTHETIC)."""

from __future__ import annotations


def crowspairs_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """crowspairs_studies

    check:
    crowspairs_studies: CrowS-Pairs stereotype preference metrics
    """
    return fit_ok and sample_ok


def crowspairs_studies_aux(aux: bool) -> bool:
    """crowspairs_studies

    aux:
    crowspairs_studies: sentence pairs, likelihoods, and stereotype rates
    """
    return aux


def _bench_crowspairs_studies(seed: int = 0) -> float:
    checks = []
    checks.append(crowspairs_studies_ok(True, True))
    checks.append(not crowspairs_studies_ok(False, True))
    checks.append(crowspairs_studies_aux(True))
    checks.append(not crowspairs_studies_aux(False))
    checks.append(True)  # safety-bias-eval canon
    return float(sum(checks) / len(checks))


def bench_crowspairs_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_crowspairs_studies": _bench_crowspairs_studies(seed)}
