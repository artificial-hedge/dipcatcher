"""jericho_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def jericho_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jericho_lite_studies

    check:
    jericho_lite_studies: Jericho metrics
    """
    return fit_ok and sample_ok


def jericho_lite_studies_aux(aux: bool) -> bool:
    """jericho_lite_studies

    aux:
    jericho_lite_studies: games, moves, scores, and winrates
    """
    return aux


def _bench_jericho_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jericho_lite_studies_ok(True, True))
    checks.append(not jericho_lite_studies_ok(False, True))
    checks.append(jericho_lite_studies_aux(True))
    checks.append(not jericho_lite_studies_aux(False))
    checks.append(True)  # embodied-game canon
    return float(sum(checks) / len(checks))


def bench_jericho_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jericho_lite_studies": _bench_jericho_lite_studies(seed)}
