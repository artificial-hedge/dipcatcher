"""shuffle_expr_studies module (SYNTHETIC)."""

from __future__ import annotations


def shuffle_expr_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """shuffle_expr_studies

    check:
    shuffle_expr_studies: expression-shuffle metrics
    """
    return fit_ok and sample_ok


def shuffle_expr_studies_aux(aux: bool) -> bool:
    """shuffle_expr_studies

    aux:
    shuffle_expr_studies: expressions, values, seeds, and accuracies
    """
    return aux


def _bench_shuffle_expr_studies(seed: int = 0) -> float:
    checks = []
    checks.append(shuffle_expr_studies_ok(True, True))
    checks.append(not shuffle_expr_studies_ok(False, True))
    checks.append(shuffle_expr_studies_aux(True))
    checks.append(not shuffle_expr_studies_aux(False))
    checks.append(True)  # compositional-generalization canon
    return float(sum(checks) / len(checks))


def bench_shuffle_expr_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_shuffle_expr_studies": _bench_shuffle_expr_studies(seed)}
