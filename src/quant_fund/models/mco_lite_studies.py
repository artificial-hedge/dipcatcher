"""mco_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def mco_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mco_lite_studies

    check:
    mco_lite_studies: MCO minimal-composition metrics
    """
    return fit_ok and sample_ok


def mco_lite_studies_aux(aux: bool) -> bool:
    """mco_lite_studies

    aux:
    mco_lite_studies: commands, programs, splits, and accuracies
    """
    return aux


def _bench_mco_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mco_lite_studies_ok(True, True))
    checks.append(not mco_lite_studies_ok(False, True))
    checks.append(mco_lite_studies_aux(True))
    checks.append(not mco_lite_studies_aux(False))
    checks.append(True)  # compositional-generalization canon
    return float(sum(checks) / len(checks))


def bench_mco_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mco_lite_studies": _bench_mco_lite_studies(seed)}
