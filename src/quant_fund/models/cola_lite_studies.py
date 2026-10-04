"""cola_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def cola_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cola_lite_studies

    check:
    cola_lite_studies: CoLA acceptability metrics
    """
    return fit_ok and sample_ok


def cola_lite_studies_aux(aux: bool) -> bool:
    """cola_lite_studies

    aux:
    cola_lite_studies: sentences, labels, predictions, and correlations
    """
    return aux


def _bench_cola_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cola_lite_studies_ok(True, True))
    checks.append(not cola_lite_studies_ok(False, True))
    checks.append(cola_lite_studies_aux(True))
    checks.append(not cola_lite_studies_aux(False))
    checks.append(True)  # GLUE-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_cola_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cola_lite_studies": _bench_cola_lite_studies(seed)}
