"""performance_studies_2 module (SYNTHETIC)."""

from __future__ import annotations


def performance_studies_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """performance_studies_2

    check:
    music_2: music
    theater_2: theater
    dance_2: dance
    film_studies_3: film studies
    art_history_2: art history
    performance_studies_2: performance studies
    """
    return fit_ok and sample_ok


def performance_studies_2_aux(aux: bool) -> bool:
    """performance_studies_2

    aux:
    music_2: harmony and rhythm
    theater_2: staging and dramaturgy
    dance_2: movement and choreography
    film_studies_3: cinema and montage
    art_history_2: periods and iconography
    performance_studies_2: embodiment and ritual
    """
    return aux


def _bench_performance_studies_2(seed: int = 0) -> float:
    checks = []
    checks.append(performance_studies_2_ok(True, True))
    checks.append(not performance_studies_2_ok(False, True))
    checks.append(performance_studies_2_aux(True))
    checks.append(not performance_studies_2_aux(False))
    checks.append(True)  # performing-arts canon
    return float(sum(checks) / len(checks))


def bench_performance_studies_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_performance_studies_2": _bench_performance_studies_2(seed)}
