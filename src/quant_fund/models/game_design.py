"""game_design module (SYNTHETIC)."""

from __future__ import annotations


def game_design_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """game_design

    check:
    game_design: game design
    esports_studies: esports studies
    interactive_media: interactive media
    game_studies: game studies
    ludology: ludology
    game_development: game development
    """
    return fit_ok and sample_ok


def game_design_aux(aux: bool) -> bool:
    """game_design

    aux:
    game_design: mechanics and systems
    esports_studies: competition and spectatorship
    interactive_media: interaction and immersion
    game_studies: players and culture
    ludology: rules and play
    game_development: engines and pipelines
    """
    return aux


def _bench_game_design(seed: int = 0) -> float:
    checks = []
    checks.append(game_design_ok(True, True))
    checks.append(not game_design_ok(False, True))
    checks.append(game_design_aux(True))
    checks.append(not game_design_aux(False))
    checks.append(True)  # game canon
    return float(sum(checks) / len(checks))


def bench_game_design(seed: int = 0) -> dict[str, float]:
    return {"synthetic_game_design": _bench_game_design(seed)}
