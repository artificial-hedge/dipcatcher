"""textworld_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def textworld_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """textworld_lite_studies

    check:
    textworld_lite_studies: TextWorld metrics
    """
    return fit_ok and sample_ok


def textworld_lite_studies_aux(aux: bool) -> bool:
    """textworld_lite_studies

    aux:
    textworld_lite_studies: quests, commands, feedback, and scores
    """
    return aux


def _bench_textworld_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(textworld_lite_studies_ok(True, True))
    checks.append(not textworld_lite_studies_ok(False, True))
    checks.append(textworld_lite_studies_aux(True))
    checks.append(not textworld_lite_studies_aux(False))
    checks.append(True)  # embodied-game canon
    return float(sum(checks) / len(checks))


def bench_textworld_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_textworld_lite_studies": _bench_textworld_lite_studies(seed)}
