"""directing_studies module (SYNTHETIC)."""

from __future__ import annotations


def directing_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """directing_studies

    check:
    performing_arts_2: performing arts
    theater_arts: theater arts
    acting_studies: acting studies
    directing_studies: directing studies
    playwriting: playwriting
    scenography: scenography
    """
    return fit_ok and sample_ok


def directing_studies_aux(aux: bool) -> bool:
    """directing_studies

    aux:
    performing_arts_2: stage and audience
    theater_arts: drama and production
    acting_studies: character and rehearsal
    directing_studies: staging and vision
    playwriting: script and dialogue
    scenography: space and design
    """
    return aux


def _bench_directing_studies(seed: int = 0) -> float:
    checks = []
    checks.append(directing_studies_ok(True, True))
    checks.append(not directing_studies_ok(False, True))
    checks.append(directing_studies_aux(True))
    checks.append(not directing_studies_aux(False))
    checks.append(True)  # performing-arts canon
    return float(sum(checks) / len(checks))


def bench_directing_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_directing_studies": _bench_directing_studies(seed)}
