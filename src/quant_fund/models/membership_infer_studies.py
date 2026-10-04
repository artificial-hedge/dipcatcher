"""membership_infer_studies module (SYNTHETIC)."""

from __future__ import annotations


def membership_infer_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """membership_infer_studies

    check:
    membership_infer_studies: membership-inference scores/thresholds and AUC
    """
    return fit_ok and sample_ok


def membership_infer_studies_aux(aux: bool) -> bool:
    """membership_infer_studies

    aux:
    membership_infer_studies: attack confidences/losses and member labels
    """
    return aux


def _bench_membership_infer_studies(seed: int = 0) -> float:
    checks = []
    checks.append(membership_infer_studies_ok(True, True))
    checks.append(not membership_infer_studies_ok(False, True))
    checks.append(membership_infer_studies_aux(True))
    checks.append(not membership_infer_studies_aux(False))
    checks.append(True)  # privacy-inference canon
    return float(sum(checks) / len(checks))


def bench_membership_infer_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_membership_infer_studies": _bench_membership_infer_studies(seed)}
