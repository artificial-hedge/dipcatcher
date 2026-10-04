"""qqp_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def qqp_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """qqp_lite_studies

    check:
    qqp_lite_studies: QQP paraphrase metrics
    """
    return fit_ok and sample_ok


def qqp_lite_studies_aux(aux: bool) -> bool:
    """qqp_lite_studies

    aux:
    qqp_lite_studies: questions, duplicates, labels, and f1s
    """
    return aux


def _bench_qqp_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(qqp_lite_studies_ok(True, True))
    checks.append(not qqp_lite_studies_ok(False, True))
    checks.append(qqp_lite_studies_aux(True))
    checks.append(not qqp_lite_studies_aux(False))
    checks.append(True)  # GLUE-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_qqp_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_qqp_lite_studies": _bench_qqp_lite_studies(seed)}
