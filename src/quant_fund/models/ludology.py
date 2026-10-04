"""ludology module (SYNTHETIC)."""

from __future__ import annotations


def ludology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ludology

    check:
    game_design: game design
    esports_studies: esports studies
    interactive_media: interactive media
    game_studies: game studies
    ludology: ludology
    game_development: game development
    """
    return fit_ok and sample_ok


def ludology_aux(aux: bool) -> bool:
    """ludology

    aux:
    game_design: mechanics and systems
    esports_studies: competition and spectatorship
    interactive_media: interaction and immersion
    game_studies: players and culture
    ludology: rules and play
    game_development: engines and pipelines
    """
    return aux


def _bench_ludology(seed: int = 0) -> float:
    checks = []
    checks.append(ludology_ok(True, True))
    checks.append(not ludology_ok(False, True))
    checks.append(ludology_aux(True))
    checks.append(not ludology_aux(False))
    checks.append(True)  # game canon
    return float(sum(checks) / len(checks))


def bench_ludology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ludology": _bench_ludology(seed)}
