"""board_game_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def board_game_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """board_game_qa_studies

    check:
    board_game_qa_studies: BoardgameQA metrics
    """
    return fit_ok and sample_ok


def board_game_qa_studies_aux(aux: bool) -> bool:
    """board_game_qa_studies

    aux:
    board_game_qa_studies: examples, rules, queries, and scores
    """
    return aux


def _bench_board_game_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(board_game_qa_studies_ok(True, True))
    checks.append(not board_game_qa_studies_ok(False, True))
    checks.append(board_game_qa_studies_aux(True))
    checks.append(not board_game_qa_studies_aux(False))
    checks.append(True)  # NLU-exotics canon
    return float(sum(checks) / len(checks))


def bench_board_game_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_board_game_qa_studies": _bench_board_game_qa_studies(seed)}
