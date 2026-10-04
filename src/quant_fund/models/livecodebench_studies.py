"""livecodebench_studies module (SYNTHETIC)."""

from __future__ import annotations


def livecodebench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """livecodebench_studies

    check:
    livecodebench_studies: LiveCodeBench contamination-free pass@1 metrics
    """
    return fit_ok and sample_ok


def livecodebench_studies_aux(aux: bool) -> bool:
    """livecodebench_studies

    aux:
    livecodebench_studies: dated tasks, tests, and pass rates
    """
    return aux


def _bench_livecodebench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(livecodebench_studies_ok(True, True))
    checks.append(not livecodebench_studies_ok(False, True))
    checks.append(livecodebench_studies_aux(True))
    checks.append(not livecodebench_studies_aux(False))
    checks.append(True)  # code-eval canon
    return float(sum(checks) / len(checks))


def bench_livecodebench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_livecodebench_studies": _bench_livecodebench_studies(seed)}
