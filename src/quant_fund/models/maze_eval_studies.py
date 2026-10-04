"""maze_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def maze_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """maze_eval_studies

    check:
    maze_eval_studies: MazeEval metrics
    """
    return fit_ok and sample_ok


def maze_eval_studies_aux(aux: bool) -> bool:
    """maze_eval_studies

    aux:
    maze_eval_studies: mazes, paths, decisions, and scores
    """
    return aux


def _bench_maze_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(maze_eval_studies_ok(True, True))
    checks.append(not maze_eval_studies_ok(False, True))
    checks.append(maze_eval_studies_aux(True))
    checks.append(not maze_eval_studies_aux(False))
    checks.append(True)  # web-agent canon
    return float(sum(checks) / len(checks))


def bench_maze_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_maze_eval_studies": _bench_maze_eval_studies(seed)}
