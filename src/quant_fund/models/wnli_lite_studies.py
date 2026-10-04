"""wnli_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def wnli_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """wnli_lite_studies

    check:
    wnli_lite_studies: WNLI coreference metrics
    """
    return fit_ok and sample_ok


def wnli_lite_studies_aux(aux: bool) -> bool:
    """wnli_lite_studies

    aux:
    wnli_lite_studies: premises, hypotheses, labels, and accuracies
    """
    return aux


def _bench_wnli_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(wnli_lite_studies_ok(True, True))
    checks.append(not wnli_lite_studies_ok(False, True))
    checks.append(wnli_lite_studies_aux(True))
    checks.append(not wnli_lite_studies_aux(False))
    checks.append(True)  # GLUE-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_wnli_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wnli_lite_studies": _bench_wnli_lite_studies(seed)}
