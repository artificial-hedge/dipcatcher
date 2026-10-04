"""uni_math_studies module (SYNTHETIC)."""

from __future__ import annotations


def uni_math_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """uni_math_studies

    check:
    uni_math_studies: University-level math metrics
    """
    return fit_ok and sample_ok


def uni_math_studies_aux(aux: bool) -> bool:
    """uni_math_studies

    aux:
    uni_math_studies: topics, problems, solutions, and pass rates
    """
    return aux


def _bench_uni_math_studies(seed: int = 0) -> float:
    checks = []
    checks.append(uni_math_studies_ok(True, True))
    checks.append(not uni_math_studies_ok(False, True))
    checks.append(uni_math_studies_aux(True))
    checks.append(not uni_math_studies_aux(False))
    checks.append(True)  # math-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_uni_math_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_uni_math_studies": _bench_uni_math_studies(seed)}
