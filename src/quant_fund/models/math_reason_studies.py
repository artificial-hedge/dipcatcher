"""math_reason_studies module (SYNTHETIC)."""

from __future__ import annotations


def math_reason_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """math_reason_studies

    check:
    math_reason_studies: Competition math-reasoning metrics
    """
    return fit_ok and sample_ok


def math_reason_studies_aux(aux: bool) -> bool:
    """math_reason_studies

    aux:
    math_reason_studies: problems, derivations, answers, and scores
    """
    return aux


def _bench_math_reason_studies(seed: int = 0) -> float:
    checks = []
    checks.append(math_reason_studies_ok(True, True))
    checks.append(not math_reason_studies_ok(False, True))
    checks.append(math_reason_studies_aux(True))
    checks.append(not math_reason_studies_aux(False))
    checks.append(True)  # math-reasoning-eval canon
    return float(sum(checks) / len(checks))


def bench_math_reason_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_math_reason_studies": _bench_math_reason_studies(seed)}
