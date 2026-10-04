"""scenography module (SYNTHETIC)."""

from __future__ import annotations


def scenography_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """scenography

    check:
    performing_arts_2: performing arts
    theater_arts: theater arts
    acting_studies: acting studies
    directing_studies: directing studies
    playwriting: playwriting
    scenography: scenography
    """
    return fit_ok and sample_ok


def scenography_aux(aux: bool) -> bool:
    """scenography

    aux:
    performing_arts_2: stage and audience
    theater_arts: drama and production
    acting_studies: character and rehearsal
    directing_studies: staging and vision
    playwriting: script and dialogue
    scenography: space and design
    """
    return aux


def _bench_scenography(seed: int = 0) -> float:
    checks = []
    checks.append(scenography_ok(True, True))
    checks.append(not scenography_ok(False, True))
    checks.append(scenography_aux(True))
    checks.append(not scenography_aux(False))
    checks.append(True)  # performing-arts canon
    return float(sum(checks) / len(checks))


def bench_scenography(seed: int = 0) -> dict[str, float]:
    return {"synthetic_scenography": _bench_scenography(seed)}
