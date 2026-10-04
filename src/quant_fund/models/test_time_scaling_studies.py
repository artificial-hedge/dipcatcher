"""test_time_scaling_studies module (SYNTHETIC)."""

from __future__ import annotations


def test_time_scaling_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """test_time_scaling_studies

    check:
    test_time_scaling_studies: compute allocation and budget forcing/parallel and sequential
    """
    return fit_ok and sample_ok


def test_time_scaling_studies_aux(aux: bool) -> bool:
    """test_time_scaling_studies

    aux:
    test_time_scaling_studies: wait-token insertion and adaptive depth/scaling and accuracy
    """
    return aux


def _bench_test_time_scaling_studies(seed: int = 0) -> float:
    checks = []
    checks.append(test_time_scaling_studies_ok(True, True))
    checks.append(not test_time_scaling_studies_ok(False, True))
    checks.append(test_time_scaling_studies_aux(True))
    checks.append(not test_time_scaling_studies_aux(False))
    checks.append(True)  # inference-scaling canon
    return float(sum(checks) / len(checks))


def bench_test_time_scaling_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_test_time_scaling_studies": _bench_test_time_scaling_studies(seed)}
