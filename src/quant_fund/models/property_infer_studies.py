"""property_infer_studies module (SYNTHETIC)."""

from __future__ import annotations


def property_infer_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """property_infer_studies

    check:
    property_infer_studies: property inference on population stats and rates
    """
    return fit_ok and sample_ok


def property_infer_studies_aux(aux: bool) -> bool:
    """property_infer_studies

    aux:
    property_infer_studies: meta-classifiers, shadow models, and AUCs
    """
    return aux


def _bench_property_infer_studies(seed: int = 0) -> float:
    checks = []
    checks.append(property_infer_studies_ok(True, True))
    checks.append(not property_infer_studies_ok(False, True))
    checks.append(property_infer_studies_aux(True))
    checks.append(not property_infer_studies_aux(False))
    checks.append(True)  # privacy-inference canon
    return float(sum(checks) / len(checks))


def bench_property_infer_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_property_infer_studies": _bench_property_infer_studies(seed)}
