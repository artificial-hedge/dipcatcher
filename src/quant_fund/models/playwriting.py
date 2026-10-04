"""playwriting module (SYNTHETIC)."""

from __future__ import annotations


def playwriting_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """playwriting

    check:
    performing_arts_2: performing arts
    theater_arts: theater arts
    acting_studies: acting studies
    directing_studies: directing studies
    playwriting: playwriting
    scenography: scenography
    """
    return fit_ok and sample_ok


def playwriting_aux(aux: bool) -> bool:
    """playwriting

    aux:
    performing_arts_2: stage and audience
    theater_arts: drama and production
    acting_studies: character and rehearsal
    directing_studies: staging and vision
    playwriting: script and dialogue
    scenography: space and design
    """
    return aux


def _bench_playwriting(seed: int = 0) -> float:
    checks = []
    checks.append(playwriting_ok(True, True))
    checks.append(not playwriting_ok(False, True))
    checks.append(playwriting_aux(True))
    checks.append(not playwriting_aux(False))
    checks.append(True)  # performing-arts canon
    return float(sum(checks) / len(checks))


def bench_playwriting(seed: int = 0) -> dict[str, float]:
    return {"synthetic_playwriting": _bench_playwriting(seed)}
