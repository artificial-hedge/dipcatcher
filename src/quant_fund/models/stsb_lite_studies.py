"""stsb_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def stsb_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """stsb_lite_studies

    check:
    stsb_lite_studies: STS-B similarity metrics
    """
    return fit_ok and sample_ok


def stsb_lite_studies_aux(aux: bool) -> bool:
    """stsb_lite_studies

    aux:
    stsb_lite_studies: sentences, scores, predictions, and correlations
    """
    return aux


def _bench_stsb_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(stsb_lite_studies_ok(True, True))
    checks.append(not stsb_lite_studies_ok(False, True))
    checks.append(stsb_lite_studies_aux(True))
    checks.append(not stsb_lite_studies_aux(False))
    checks.append(True)  # GLUE-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_stsb_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stsb_lite_studies": _bench_stsb_lite_studies(seed)}
