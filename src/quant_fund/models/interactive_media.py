"""interactive_media module (SYNTHETIC)."""

from __future__ import annotations


def interactive_media_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """interactive_media

    check:
    game_design: game design
    esports_studies: esports studies
    interactive_media: interactive media
    game_studies: game studies
    ludology: ludology
    game_development: game development
    """
    return fit_ok and sample_ok


def interactive_media_aux(aux: bool) -> bool:
    """interactive_media

    aux:
    game_design: mechanics and systems
    esports_studies: competition and spectatorship
    interactive_media: interaction and immersion
    game_studies: players and culture
    ludology: rules and play
    game_development: engines and pipelines
    """
    return aux


def _bench_interactive_media(seed: int = 0) -> float:
    checks = []
    checks.append(interactive_media_ok(True, True))
    checks.append(not interactive_media_ok(False, True))
    checks.append(interactive_media_aux(True))
    checks.append(not interactive_media_aux(False))
    checks.append(True)  # game canon
    return float(sum(checks) / len(checks))


def bench_interactive_media(seed: int = 0) -> dict[str, float]:
    return {"synthetic_interactive_media": _bench_interactive_media(seed)}
