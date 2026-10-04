"""performing_arts_2 module (SYNTHETIC)."""

from __future__ import annotations


def performing_arts_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """performing_arts_2

    check:
    performing_arts_2: performing arts
    theater_arts: theater arts
    acting_studies: acting studies
    directing_studies: directing studies
    playwriting: playwriting
    scenography: scenography
    """
    return fit_ok and sample_ok


def performing_arts_2_aux(aux: bool) -> bool:
    """performing_arts_2

    aux:
    performing_arts_2: stage and audience
    theater_arts: drama and production
    acting_studies: character and rehearsal
    directing_studies: staging and vision
    playwriting: script and dialogue
    scenography: space and design
    """
    return aux


def _bench_performing_arts_2(seed: int = 0) -> float:
    checks = []
    checks.append(performing_arts_2_ok(True, True))
    checks.append(not performing_arts_2_ok(False, True))
    checks.append(performing_arts_2_aux(True))
    checks.append(not performing_arts_2_aux(False))
    checks.append(True)  # performing-arts canon
    return float(sum(checks) / len(checks))


def bench_performing_arts_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_performing_arts_2": _bench_performing_arts_2(seed)}
