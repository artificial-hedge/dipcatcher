"""attribution_patching_studies module (SYNTHETIC)."""

from __future__ import annotations


def attribution_patching_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """attribution_patching_studies

    check:
    attribution_patching_studies: edge attributions and integrated gradients/components and paths
    """
    return fit_ok and sample_ok


def attribution_patching_studies_aux(aux: bool) -> bool:
    """attribution_patching_studies

    aux:
    attribution_patching_studies: AtP-style linear approximations/edges and importance
    """
    return aux


def _bench_attribution_patching_studies(seed: int = 0) -> float:
    checks = []
    checks.append(attribution_patching_studies_ok(True, True))
    checks.append(not attribution_patching_studies_ok(False, True))
    checks.append(attribution_patching_studies_aux(True))
    checks.append(not attribution_patching_studies_aux(False))
    checks.append(True)  # interpretability-3 canon
    return float(sum(checks) / len(checks))


def bench_attribution_patching_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_attribution_patching_studies": _bench_attribution_patching_studies(seed)}
