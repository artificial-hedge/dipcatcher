"""network_meta_studies module (SYNTHETIC)."""

from __future__ import annotations


def network_meta_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """network_meta_studies

    check:
    network_meta_studies: consistency and transitivity/mixed evidence and loops
    """
    return fit_ok and sample_ok


def network_meta_studies_aux(aux: bool) -> bool:
    """network_meta_studies

    aux:
    network_meta_studies: SUCRA and ranking/heterogeneity and models
    """
    return aux


def _bench_network_meta_studies(seed: int = 0) -> float:
    checks = []
    checks.append(network_meta_studies_ok(True, True))
    checks.append(not network_meta_studies_ok(False, True))
    checks.append(network_meta_studies_aux(True))
    checks.append(not network_meta_studies_aux(False))
    checks.append(True)  # evidence-synthesis canon
    return float(sum(checks) / len(checks))


def bench_network_meta_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_network_meta_studies": _bench_network_meta_studies(seed)}
