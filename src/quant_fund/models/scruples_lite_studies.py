"""scruples_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def scruples_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """scruples_lite_studies

    check:
    scruples_lite_studies: SCRUPLES moral-dilemma metrics
    """
    return fit_ok and sample_ok


def scruples_lite_studies_aux(aux: bool) -> bool:
    """scruples_lite_studies

    aux:
    scruples_lite_studies: dilemmas, actions, labels, and accuracies
    """
    return aux


def _bench_scruples_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(scruples_lite_studies_ok(True, True))
    checks.append(not scruples_lite_studies_ok(False, True))
    checks.append(scruples_lite_studies_aux(True))
    checks.append(not scruples_lite_studies_aux(False))
    checks.append(True)  # ethics-eval canon
    return float(sum(checks) / len(checks))


def bench_scruples_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_scruples_lite_studies": _bench_scruples_lite_studies(seed)}
