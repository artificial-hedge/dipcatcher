"""benchmark_gaming_studies module (SYNTHETIC)."""

from __future__ import annotations


def benchmark_gaming_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """benchmark_gaming_studies

    check:
    benchmark_gaming_studies: benchmark-gaming detector rules/scores and deltas
    """
    return fit_ok and sample_ok


def benchmark_gaming_studies_aux(aux: bool) -> bool:
    """benchmark_gaming_studies

    aux:
    benchmark_gaming_studies: gaming-susceptible probe suites/inputs and artifacts
    """
    return aux


def _bench_benchmark_gaming_studies(seed: int = 0) -> float:
    checks = []
    checks.append(benchmark_gaming_studies_ok(True, True))
    checks.append(not benchmark_gaming_studies_ok(False, True))
    checks.append(benchmark_gaming_studies_aux(True))
    checks.append(not benchmark_gaming_studies_aux(False))
    checks.append(True)  # eval-science canon
    return float(sum(checks) / len(checks))


def bench_benchmark_gaming_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_benchmark_gaming_studies": _bench_benchmark_gaming_studies(seed)}
