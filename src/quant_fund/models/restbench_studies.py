"""restbench_studies module (SYNTHETIC)."""

from __future__ import annotations


def restbench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """restbench_studies

    check:
    restbench_studies: RestBench metrics
    """
    return fit_ok and sample_ok


def restbench_studies_aux(aux: bool) -> bool:
    """restbench_studies

    aux:
    restbench_studies: apis, tasks, traces, and scores
    """
    return aux


def _bench_restbench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(restbench_studies_ok(True, True))
    checks.append(not restbench_studies_ok(False, True))
    checks.append(restbench_studies_aux(True))
    checks.append(not restbench_studies_aux(False))
    checks.append(True)  # code-agent canon
    return float(sum(checks) / len(checks))


def bench_restbench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_restbench_studies": _bench_restbench_studies(seed)}
