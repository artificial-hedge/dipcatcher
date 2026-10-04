"""tab_math_studies module (SYNTHETIC)."""

from __future__ import annotations


def tab_math_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tab_math_studies

    check:
    tab_math_studies: Table-math reasoning metrics
    """
    return fit_ok and sample_ok


def tab_math_studies_aux(aux: bool) -> bool:
    """tab_math_studies

    aux:
    tab_math_studies: tables, questions, derivations, and accuracies
    """
    return aux


def _bench_tab_math_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tab_math_studies_ok(True, True))
    checks.append(not tab_math_studies_ok(False, True))
    checks.append(tab_math_studies_aux(True))
    checks.append(not tab_math_studies_aux(False))
    checks.append(True)  # math-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_tab_math_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tab_math_studies": _bench_tab_math_studies(seed)}
