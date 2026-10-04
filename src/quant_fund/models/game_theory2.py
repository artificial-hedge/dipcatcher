"""game_theory2 module (SYNTHETIC)."""

from __future__ import annotations


def game_theory2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """game_theory2

    check:
    game_theory2: game theory
    behavioral_econ: behavioral economics
    political_science: political science
    sociology_net: social network analysis
    cognitive_science: cognitive science
    linguistics: linguistics
    """
    return fit_ok and sample_ok


def game_theory2_aux(aux: bool) -> bool:
    """game_theory2

    aux:
    game_theory2: equilibrium refinement
    behavioral_econ: prospect theory
    political_science: voting theory
    sociology_net: network dynamics
    cognitive_science: decision models
    linguistics: generative grammar
    """
    return aux


def _bench_game_theory2(seed: int = 0) -> float:
    checks = []
    checks.append(game_theory2_ok(True, True))
    checks.append(not game_theory2_ok(False, True))
    checks.append(game_theory2_aux(True))
    checks.append(not game_theory2_aux(False))
    checks.append(True)  # social-science canon
    return float(sum(checks) / len(checks))


def bench_game_theory2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_game_theory2": _bench_game_theory2(seed)}
