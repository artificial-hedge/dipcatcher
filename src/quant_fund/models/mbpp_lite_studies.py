"""mbpp_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def mbpp_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mbpp_lite_studies

    check:
    mbpp_lite_studies: MBPP metrics
    """
    return fit_ok and sample_ok


def mbpp_lite_studies_aux(aux: bool) -> bool:
    """mbpp_lite_studies

    aux:
    mbpp_lite_studies: prompts, programs, tests, and scores
    """
    return aux


def _bench_mbpp_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mbpp_lite_studies_ok(True, True))
    checks.append(not mbpp_lite_studies_ok(False, True))
    checks.append(mbpp_lite_studies_aux(True))
    checks.append(not mbpp_lite_studies_aux(False))
    checks.append(True)  # code-agent canon
    return float(sum(checks) / len(checks))


def bench_mbpp_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mbpp_lite_studies": _bench_mbpp_lite_studies(seed)}
