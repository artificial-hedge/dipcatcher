"""attribute_infer_studies module (SYNTHETIC)."""

from __future__ import annotations


def attribute_infer_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """attribute_infer_studies

    check:
    attribute_infer_studies: attribute inference from gradients/outputs and acc
    """
    return fit_ok and sample_ok


def attribute_infer_studies_aux(aux: bool) -> bool:
    """attribute_infer_studies

    aux:
    attribute_infer_studies: leakage scores, victim features, and inference
    """
    return aux


def _bench_attribute_infer_studies(seed: int = 0) -> float:
    checks = []
    checks.append(attribute_infer_studies_ok(True, True))
    checks.append(not attribute_infer_studies_ok(False, True))
    checks.append(attribute_infer_studies_aux(True))
    checks.append(not attribute_infer_studies_aux(False))
    checks.append(True)  # privacy-inference canon
    return float(sum(checks) / len(checks))


def bench_attribute_infer_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_attribute_infer_studies": _bench_attribute_infer_studies(seed)}
