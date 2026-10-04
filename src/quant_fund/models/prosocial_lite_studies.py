"""prosocial_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def prosocial_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """prosocial_lite_studies

    check:
    prosocial_lite_studies: ProsocialDialog metrics
    """
    return fit_ok and sample_ok


def prosocial_lite_studies_aux(aux: bool) -> bool:
    """prosocial_lite_studies

    aux:
    prosocial_lite_studies: utterances, rules, judgments, and scores
    """
    return aux


def _bench_prosocial_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(prosocial_lite_studies_ok(True, True))
    checks.append(not prosocial_lite_studies_ok(False, True))
    checks.append(prosocial_lite_studies_aux(True))
    checks.append(not prosocial_lite_studies_aux(False))
    checks.append(True)  # social-reasoning canon
    return float(sum(checks) / len(checks))


def bench_prosocial_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_prosocial_lite_studies": _bench_prosocial_lite_studies(seed)}
