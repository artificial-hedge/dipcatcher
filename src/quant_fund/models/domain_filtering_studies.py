"""domain_filtering_studies module (SYNTHETIC)."""

from __future__ import annotations


def domain_filtering_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """domain_filtering_studies

    check:
    domain_filtering_studies: blocklist and toxic-content filtering/lists and rates
    """
    return fit_ok and sample_ok


def domain_filtering_studies_aux(aux: bool) -> bool:
    """domain_filtering_studies

    aux:
    domain_filtering_studies: language identification and source routing/detectors and routing
    """
    return aux


def _bench_domain_filtering_studies(seed: int = 0) -> float:
    checks = []
    checks.append(domain_filtering_studies_ok(True, True))
    checks.append(not domain_filtering_studies_ok(False, True))
    checks.append(domain_filtering_studies_aux(True))
    checks.append(not domain_filtering_studies_aux(False))
    checks.append(True)  # pretraining-data canon
    return float(sum(checks) / len(checks))


def bench_domain_filtering_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_domain_filtering_studies": _bench_domain_filtering_studies(seed)}
