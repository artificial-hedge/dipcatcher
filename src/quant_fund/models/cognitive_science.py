"""cognitive_science module (SYNTHETIC)."""

from __future__ import annotations


def cognitive_science_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cognitive_science

    check:
    game_theory2: game theory
    behavioral_econ: behavioral economics
    political_science: political science
    sociology_net: social network analysis
    cognitive_science: cognitive science
    linguistics: linguistics
    """
    return fit_ok and sample_ok


def cognitive_science_aux(aux: bool) -> bool:
    """cognitive_science

    aux:
    game_theory2: equilibrium refinement
    behavioral_econ: prospect theory
    political_science: voting theory
    sociology_net: network dynamics
    cognitive_science: decision models
    linguistics: generative grammar
    """
    return aux


def _bench_cognitive_science(seed: int = 0) -> float:
    checks = []
    checks.append(cognitive_science_ok(True, True))
    checks.append(not cognitive_science_ok(False, True))
    checks.append(cognitive_science_aux(True))
    checks.append(not cognitive_science_aux(False))
    checks.append(True)  # social-science canon
    return float(sum(checks) / len(checks))


def bench_cognitive_science(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cognitive_science": _bench_cognitive_science(seed)}
