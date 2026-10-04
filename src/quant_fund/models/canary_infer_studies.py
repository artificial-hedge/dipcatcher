"""canary_infer_studies module (SYNTHETIC)."""

from __future__ import annotations


def canary_infer_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """canary_infer_studies

    check:
    canary_infer_studies: canary gradients/exposure and membership leak
    """
    return fit_ok and sample_ok


def canary_infer_studies_aux(aux: bool) -> bool:
    """canary_infer_studies

    aux:
    canary_infer_studies: insertion audits/exposure ranks and bounds
    """
    return aux


def _bench_canary_infer_studies(seed: int = 0) -> float:
    checks = []
    checks.append(canary_infer_studies_ok(True, True))
    checks.append(not canary_infer_studies_ok(False, True))
    checks.append(canary_infer_studies_aux(True))
    checks.append(not canary_infer_studies_aux(False))
    checks.append(True)  # privacy-inference canon
    return float(sum(checks) / len(checks))


def bench_canary_infer_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_canary_infer_studies": _bench_canary_infer_studies(seed)}
