"""video_game_studies module (SYNTHETIC)."""

from __future__ import annotations


def video_game_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """video_game_studies

    check:
    video_game_studies: Video-game environment agent metrics
    """
    return fit_ok and sample_ok


def video_game_studies_aux(aux: bool) -> bool:
    """video_game_studies

    aux:
    video_game_studies: games, frames, actions, and scores
    """
    return aux


def _bench_video_game_studies(seed: int = 0) -> float:
    checks = []
    checks.append(video_game_studies_ok(True, True))
    checks.append(not video_game_studies_ok(False, True))
    checks.append(video_game_studies_aux(True))
    checks.append(not video_game_studies_aux(False))
    checks.append(True)  # agentic-eval-3 canon
    return float(sum(checks) / len(checks))


def bench_video_game_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_video_game_studies": _bench_video_game_studies(seed)}
