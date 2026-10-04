"""feature_geometry_studies module (SYNTHETIC)."""

from __future__ import annotations


def feature_geometry_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """feature_geometry_studies

    check:
    feature_geometry_studies: superposition and polysemanticity/directions and interference
    """
    return fit_ok and sample_ok


def feature_geometry_studies_aux(aux: bool) -> bool:
    """feature_geometry_studies

    aux:
    feature_geometry_studies: toy models and feature manifolds/clustering and sparsity
    """
    return aux


def _bench_feature_geometry_studies(seed: int = 0) -> float:
    checks = []
    checks.append(feature_geometry_studies_ok(True, True))
    checks.append(not feature_geometry_studies_ok(False, True))
    checks.append(feature_geometry_studies_aux(True))
    checks.append(not feature_geometry_studies_aux(False))
    checks.append(True)  # mech-interp-2 canon
    return float(sum(checks) / len(checks))


def bench_feature_geometry_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_feature_geometry_studies": _bench_feature_geometry_studies(seed)}
