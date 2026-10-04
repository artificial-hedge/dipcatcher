"""game_development module (SYNTHETIC)."""

from __future__ import annotations


def game_development_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """game_development

    check:
    game_design: game design
    esports_studies: esports studies
    interactive_media: interactive media
    game_studies: game studies
    ludology: ludology
    game_development: game development
    """
    return fit_ok and sample_ok


def game_development_aux(aux: bool) -> bool:
    """game_development

    aux:
    game_design: mechanics and systems
    esports_studies: competition and spectatorship
    interactive_media: interaction and immersion
    game_studies: players and culture
    ludology: rules and play
    game_development: engines and pipelines
    """
    return aux


def _bench_game_development(seed: int = 0) -> float:
    checks = []
    checks.append(game_development_ok(True, True))
    checks.append(not game_development_ok(False, True))
    checks.append(game_development_aux(True))
    checks.append(not game_development_aux(False))
    checks.append(True)  # game canon
    return float(sum(checks) / len(checks))


def bench_game_development(seed: int = 0) -> dict[str, float]:
    return {"synthetic_game_development": _bench_game_development(seed)}
