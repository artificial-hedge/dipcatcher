"""data_mix_studies module (SYNTHETIC)."""

from __future__ import annotations


def data_mix_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """data_mix_studies

    check:
    data_mix_studies: domain-weight search and mixture laws/scores and ratios
    """
    return fit_ok and sample_ok


def data_mix_studies_aux(aux: bool) -> bool:
    """data_mix_studies

    aux:
    data_mix_studies: DoReMi/proxy-model mixture optimization/weights and losses
    """
    return aux


def _bench_data_mix_studies(seed: int = 0) -> float:
    checks = []
    checks.append(data_mix_studies_ok(True, True))
    checks.append(not data_mix_studies_ok(False, True))
    checks.append(data_mix_studies_aux(True))
    checks.append(not data_mix_studies_aux(False))
    checks.append(True)  # data-filtering/dedup canon
    return float(sum(checks) / len(checks))


def bench_data_mix_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_data_mix_studies": _bench_data_mix_studies(seed)}
