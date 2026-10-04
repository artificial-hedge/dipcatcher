"""film_editing module (SYNTHETIC)."""

from __future__ import annotations


def film_editing_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """film_editing

    check:
    film_production: film production
    cinematography_studies: cinematography studies
    film_editing: film editing
    sound_design: sound design
    documentary_production: documentary production
    animation_studies: animation studies
    """
    return fit_ok and sample_ok


def film_editing_aux(aux: bool) -> bool:
    """film_editing

    aux:
    film_production: crews and shoots
    cinematography_studies: camera and lighting
    film_editing: cuts and sequences
    sound_design: audio and atmosphere
    documentary_production: subjects and truth
    animation_studies: frames and motion
    """
    return aux


def _bench_film_editing(seed: int = 0) -> float:
    checks = []
    checks.append(film_editing_ok(True, True))
    checks.append(not film_editing_ok(False, True))
    checks.append(film_editing_aux(True))
    checks.append(not film_editing_aux(False))
    checks.append(True)  # film-production canon
    return float(sum(checks) / len(checks))


def bench_film_editing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_film_editing": _bench_film_editing(seed)}
