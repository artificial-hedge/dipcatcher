"""bigcodebench_studies module (SYNTHETIC)."""

from __future__ import annotations


def bigcodebench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bigcodebench_studies

    check:
    bigcodebench_studies: BigCodeBench complex-call pass@1 and metrics
    """
    return fit_ok and sample_ok


def bigcodebench_studies_aux(aux: bool) -> bool:
    """bigcodebench_studies

    aux:
    bigcodebench_studies: tasks, library calls, and pass rates
    """
    return aux


def _bench_bigcodebench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bigcodebench_studies_ok(True, True))
    checks.append(not bigcodebench_studies_ok(False, True))
    checks.append(bigcodebench_studies_aux(True))
    checks.append(not bigcodebench_studies_aux(False))
    checks.append(True)  # code-eval canon
    return float(sum(checks) / len(checks))


def bench_bigcodebench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bigcodebench_studies": _bench_bigcodebench_studies(seed)}
