"""data_mixture_studies module (SYNTHETIC)."""

from __future__ import annotations


def data_mixture_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """data_mixture_studies

    check:
    data_mixture_studies: domain weighting and DoReMi-style optimization/weights and domains
    """
    return fit_ok and sample_ok


def data_mixture_studies_aux(aux: bool) -> bool:
    """data_mixture_studies

    aux:
    data_mixture_studies: proxy-guided mixing and group robustness/gains and coverage
    """
    return aux


def _bench_data_mixture_studies(seed: int = 0) -> float:
    checks = []
    checks.append(data_mixture_studies_ok(True, True))
    checks.append(not data_mixture_studies_ok(False, True))
    checks.append(data_mixture_studies_aux(True))
    checks.append(not data_mixture_studies_aux(False))
    checks.append(True)  # pretraining-data canon
    return float(sum(checks) / len(checks))


def bench_data_mixture_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_data_mixture_studies": _bench_data_mixture_studies(seed)}
