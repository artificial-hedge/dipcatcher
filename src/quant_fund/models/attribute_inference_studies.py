"""attribute_inference_studies module (SYNTHETIC)."""

from __future__ import annotations


def attribute_inference_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """attribute_inference_studies

    check:
    attribute_inference_studies: Attribute-inference leakage advantage metrics
    """
    return fit_ok and sample_ok


def attribute_inference_studies_aux(aux: bool) -> bool:
    """attribute_inference_studies

    aux:
    attribute_inference_studies: attributes, predictions, and leakage scores
    """
    return aux


def _bench_attribute_inference_studies(seed: int = 0) -> float:
    checks = []
    checks.append(attribute_inference_studies_ok(True, True))
    checks.append(not attribute_inference_studies_ok(False, True))
    checks.append(attribute_inference_studies_aux(True))
    checks.append(not attribute_inference_studies_aux(False))
    checks.append(True)  # privacy-inference-2 canon
    return float(sum(checks) / len(checks))


def bench_attribute_inference_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_attribute_inference_studies": _bench_attribute_inference_studies(seed)}
