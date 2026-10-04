"""game_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def game_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """game_qa_studies

    check:
    game_qa_studies: GameQA metrics
    """
    return fit_ok and sample_ok


def game_qa_studies_aux(aux: bool) -> bool:
    """game_qa_studies

    aux:
    game_qa_studies: games, rules, answers, and scores
    """
    return aux


def _bench_game_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(game_qa_studies_ok(True, True))
    checks.append(not game_qa_studies_ok(False, True))
    checks.append(game_qa_studies_aux(True))
    checks.append(not game_qa_studies_aux(False))
    checks.append(True)  # leisure canon
    return float(sum(checks) / len(checks))


def bench_game_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_game_qa_studies": _bench_game_qa_studies(seed)}
