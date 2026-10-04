"""mintaka_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def mintaka_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mintaka_lite_studies

    check:
    mintaka_lite_studies: Mintaka metrics
    """
    return fit_ok and sample_ok


def mintaka_lite_studies_aux(aux: bool) -> bool:
    """mintaka_lite_studies

    aux:
    mintaka_lite_studies: questions, entities, answers, and scores
    """
    return aux


def _bench_mintaka_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mintaka_lite_studies_ok(True, True))
    checks.append(not mintaka_lite_studies_ok(False, True))
    checks.append(mintaka_lite_studies_aux(True))
    checks.append(not mintaka_lite_studies_aux(False))
    checks.append(True)  # QA-exotics-2 canon
    return float(sum(checks) / len(checks))


def bench_mintaka_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mintaka_lite_studies": _bench_mintaka_lite_studies(seed)}
