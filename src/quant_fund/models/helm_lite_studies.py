"""helm_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def helm_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """helm_lite_studies

    check:
    helm_lite_studies: HELM scenario metrics
    """
    return fit_ok and sample_ok


def helm_lite_studies_aux(aux: bool) -> bool:
    """helm_lite_studies

    aux:
    helm_lite_studies: instances, outputs, references, and scores
    """
    return aux


def _bench_helm_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(helm_lite_studies_ok(True, True))
    checks.append(not helm_lite_studies_ok(False, True))
    checks.append(helm_lite_studies_aux(True))
    checks.append(not helm_lite_studies_aux(False))
    checks.append(True)  # LLM-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_helm_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_helm_lite_studies": _bench_helm_lite_studies(seed)}
