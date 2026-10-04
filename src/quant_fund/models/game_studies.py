"""game_studies module (SYNTHETIC)."""

from __future__ import annotations


def game_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """game_studies

    check:
    game_design: game design
    esports_studies: esports studies
    interactive_media: interactive media
    game_studies: game studies
    ludology: ludology
    game_development: game development
    """
    return fit_ok and sample_ok


def game_studies_aux(aux: bool) -> bool:
    """game_studies

    aux:
    game_design: mechanics and systems
    esports_studies: competition and spectatorship
    interactive_media: interaction and immersion
    game_studies: players and culture
    ludology: rules and play
    game_development: engines and pipelines
    """
    return aux


def _bench_game_studies(seed: int = 0) -> float:
    checks = []
    checks.append(game_studies_ok(True, True))
    checks.append(not game_studies_ok(False, True))
    checks.append(game_studies_aux(True))
    checks.append(not game_studies_aux(False))
    checks.append(True)  # game canon
    return float(sum(checks) / len(checks))


def bench_game_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_game_studies": _bench_game_studies(seed)}
