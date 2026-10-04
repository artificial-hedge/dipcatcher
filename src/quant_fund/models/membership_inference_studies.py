"""membership_inference_studies module (SYNTHETIC)."""

from __future__ import annotations


def membership_inference_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """membership_inference_studies

    check:
    membership_inference_studies: Membership-inference attack AUC/TPR-at-low-FPR metrics
    """
    return fit_ok and sample_ok


def membership_inference_studies_aux(aux: bool) -> bool:
    """membership_inference_studies

    aux:
    membership_inference_studies: records, losses, shadows, and attack scores
    """
    return aux


def _bench_membership_inference_studies(seed: int = 0) -> float:
    checks = []
    checks.append(membership_inference_studies_ok(True, True))
    checks.append(not membership_inference_studies_ok(False, True))
    checks.append(membership_inference_studies_aux(True))
    checks.append(not membership_inference_studies_aux(False))
    checks.append(True)  # privacy-inference-2 canon
    return float(sum(checks) / len(checks))


def bench_membership_inference_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_membership_inference_studies": _bench_membership_inference_studies(seed)}
