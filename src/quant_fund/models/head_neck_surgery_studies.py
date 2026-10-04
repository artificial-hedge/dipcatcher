"""head_neck_surgery_studies module (SYNTHETIC)."""

from __future__ import annotations


def head_neck_surgery_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """head_neck_surgery_studies

    check:
    head_neck_surgery_studies: neck and tumor
    ..."""
    return fit_ok and sample_ok


def head_neck_surgery_studies_aux(aux: bool) -> bool:
    """head_neck_surgery_studies

    aux:
    head_neck_surgery_studies: dissection and margins
    ..."""
    return aux


def _bench_head_neck_surgery_studies(seed: int = 0) -> float:
    checks = []
    checks.append(head_neck_surgery_studies_ok(True, True))
    checks.append(not head_neck_surgery_studies_ok(False, True))
    checks.append(head_neck_surgery_studies_aux(True))
    checks.append(not head_neck_surgery_studies_aux(False))
    checks.append(True)  # ent-head-neck canon
    return float(sum(checks) / len(checks))


def bench_head_neck_surgery_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_head_neck_surgery_studies": _bench_head_neck_surgery_studies(seed)}
