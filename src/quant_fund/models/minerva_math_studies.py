"""minerva_math_studies module (SYNTHETIC)."""

from __future__ import annotations


def minerva_math_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """minerva_math_studies

    check:
    minerva_math_studies: Minerva-style math metrics
    """
    return fit_ok and sample_ok


def minerva_math_studies_aux(aux: bool) -> bool:
    """minerva_math_studies

    aux:
    minerva_math_studies: problems, derivations, answers, and scores
    """
    return aux


def _bench_minerva_math_studies(seed: int = 0) -> float:
    checks = []
    checks.append(minerva_math_studies_ok(True, True))
    checks.append(not minerva_math_studies_ok(False, True))
    checks.append(minerva_math_studies_aux(True))
    checks.append(not minerva_math_studies_aux(False))
    checks.append(True)  # math-word-problem canon
    return float(sum(checks) / len(checks))


def bench_minerva_math_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_minerva_math_studies": _bench_minerva_math_studies(seed)}
