"""stabletoolbench_studies module (SYNTHETIC)."""

from __future__ import annotations


def stabletoolbench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """stabletoolbench_studies

    check:
    stabletoolbench_studies: StableToolBench metrics
    """
    return fit_ok and sample_ok


def stabletoolbench_studies_aux(aux: bool) -> bool:
    """stabletoolbench_studies

    aux:
    stabletoolbench_studies: tasks, solvers, failures, and scores
    """
    return aux


def _bench_stabletoolbench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(stabletoolbench_studies_ok(True, True))
    checks.append(not stabletoolbench_studies_ok(False, True))
    checks.append(stabletoolbench_studies_aux(True))
    checks.append(not stabletoolbench_studies_aux(False))
    checks.append(True)  # tool-use canon
    return float(sum(checks) / len(checks))


def bench_stabletoolbench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stabletoolbench_studies": _bench_stabletoolbench_studies(seed)}
