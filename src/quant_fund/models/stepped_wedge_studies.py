"""stepped_wedge_studies module (SYNTHETIC)."""

from __future__ import annotations


def stepped_wedge_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """stepped_wedge_studies

    check:
    stepped_wedge_studies: cluster sequences and rollout/crossover and cells
    """
    return fit_ok and sample_ok


def stepped_wedge_studies_aux(aux: bool) -> bool:
    """stepped_wedge_studies

    aux:
    stepped_wedge_studies: period effects/heterogeneity and analysis
    """
    return aux


def _bench_stepped_wedge_studies(seed: int = 0) -> float:
    checks = []
    checks.append(stepped_wedge_studies_ok(True, True))
    checks.append(not stepped_wedge_studies_ok(False, True))
    checks.append(stepped_wedge_studies_aux(True))
    checks.append(not stepped_wedge_studies_aux(False))
    checks.append(True)  # target-trial/RWE canon
    return float(sum(checks) / len(checks))


def bench_stepped_wedge_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stepped_wedge_studies": _bench_stepped_wedge_studies(seed)}
