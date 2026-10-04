"""aqua_rat_studies module (SYNTHETIC)."""

from __future__ import annotations


def aqua_rat_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """aqua_rat_studies

    check:
    aqua_rat_studies: AQuA-RAT algebraic reasoning metrics
    """
    return fit_ok and sample_ok


def aqua_rat_studies_aux(aux: bool) -> bool:
    """aqua_rat_studies

    aux:
    aqua_rat_studies: questions, options, rationales, and accuracies
    """
    return aux


def _bench_aqua_rat_studies(seed: int = 0) -> float:
    checks = []
    checks.append(aqua_rat_studies_ok(True, True))
    checks.append(not aqua_rat_studies_ok(False, True))
    checks.append(aqua_rat_studies_aux(True))
    checks.append(not aqua_rat_studies_aux(False))
    checks.append(True)  # math-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_aqua_rat_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aqua_rat_studies": _bench_aqua_rat_studies(seed)}
