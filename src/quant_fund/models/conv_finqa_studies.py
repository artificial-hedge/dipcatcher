"""conv_finqa_studies module (SYNTHETIC)."""

from __future__ import annotations


def conv_finqa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """conv_finqa_studies

    check:
    conv_finqa_studies: ConvFinQA metrics
    """
    return fit_ok and sample_ok


def conv_finqa_studies_aux(aux: bool) -> bool:
    """conv_finqa_studies

    aux:
    conv_finqa_studies: reports, turns, programs, and scores
    """
    return aux


def _bench_conv_finqa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(conv_finqa_studies_ok(True, True))
    checks.append(not conv_finqa_studies_ok(False, True))
    checks.append(conv_finqa_studies_aux(True))
    checks.append(not conv_finqa_studies_aux(False))
    checks.append(True)  # NLU-exotics canon
    return float(sum(checks) / len(checks))


def bench_conv_finqa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_conv_finqa_studies": _bench_conv_finqa_studies(seed)}
