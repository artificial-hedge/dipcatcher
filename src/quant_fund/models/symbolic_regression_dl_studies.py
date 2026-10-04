"""symbolic_regression_dl_studies module (SYNTHETIC)."""

from __future__ import annotations


def symbolic_regression_dl_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """symbolic_regression_dl_studies

    check:
    symbolic_regression_dl_studies: expression trees and equation discovery/fit and complexity
    """
    return fit_ok and sample_ok


def symbolic_regression_dl_studies_aux(aux: bool) -> bool:
    """symbolic_regression_dl_studies

    aux:
    symbolic_regression_dl_studies: deep generation and token sequences/validity and simplicity
    """
    return aux


def _bench_symbolic_regression_dl_studies(seed: int = 0) -> float:
    checks = []
    checks.append(symbolic_regression_dl_studies_ok(True, True))
    checks.append(not symbolic_regression_dl_studies_ok(False, True))
    checks.append(symbolic_regression_dl_studies_aux(True))
    checks.append(not symbolic_regression_dl_studies_aux(False))
    checks.append(True)  # neuro-symbolic canon
    return float(sum(checks) / len(checks))


def bench_symbolic_regression_dl_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_symbolic_regression_dl_studies": _bench_symbolic_regression_dl_studies(seed)}
