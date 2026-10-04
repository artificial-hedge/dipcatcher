"""drop_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def drop_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """drop_lite_studies

    check:
    drop_lite_studies: DROP discrete-reasoning metrics
    """
    return fit_ok and sample_ok


def drop_lite_studies_aux(aux: bool) -> bool:
    """drop_lite_studies

    aux:
    drop_lite_studies: passages, questions, answers, and accuracies
    """
    return aux


def _bench_drop_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(drop_lite_studies_ok(True, True))
    checks.append(not drop_lite_studies_ok(False, True))
    checks.append(drop_lite_studies_aux(True))
    checks.append(not drop_lite_studies_aux(False))
    checks.append(True)  # reading-comp-4 canon
    return float(sum(checks) / len(checks))


def bench_drop_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_drop_lite_studies": _bench_drop_lite_studies(seed)}
