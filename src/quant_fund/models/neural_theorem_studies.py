"""neural_theorem_studies module (SYNTHETIC)."""

from __future__ import annotations


def neural_theorem_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """neural_theorem_studies

    check:
    neural_theorem_studies: tactic prediction and proof search/Lean and Isabelle
    """
    return fit_ok and sample_ok


def neural_theorem_studies_aux(aux: bool) -> bool:
    """neural_theorem_studies

    aux:
    neural_theorem_studies: premise retrieval and expression trees/goals and state
    """
    return aux


def _bench_neural_theorem_studies(seed: int = 0) -> float:
    checks = []
    checks.append(neural_theorem_studies_ok(True, True))
    checks.append(not neural_theorem_studies_ok(False, True))
    checks.append(neural_theorem_studies_aux(True))
    checks.append(not neural_theorem_studies_aux(False))
    checks.append(True)  # neuro-symbolic canon
    return float(sum(checks) / len(checks))


def bench_neural_theorem_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_neural_theorem_studies": _bench_neural_theorem_studies(seed)}
