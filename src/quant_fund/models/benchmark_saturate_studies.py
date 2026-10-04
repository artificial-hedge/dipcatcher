"""benchmark_saturate_studies module (SYNTHETIC)."""

from __future__ import annotations


def benchmark_saturate_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """benchmark_saturate_studies

    check:
    benchmark_saturate_studies: benchmark saturation-ceiling metrics/scores and heads
    """
    return fit_ok and sample_ok


def benchmark_saturate_studies_aux(aux: bool) -> bool:
    """benchmark_saturate_studies

    aux:
    benchmark_saturate_studies: saturated-vs-active task splits/benchmarks and accs
    """
    return aux


def _bench_benchmark_saturate_studies(seed: int = 0) -> float:
    checks = []
    checks.append(benchmark_saturate_studies_ok(True, True))
    checks.append(not benchmark_saturate_studies_ok(False, True))
    checks.append(benchmark_saturate_studies_aux(True))
    checks.append(not benchmark_saturate_studies_aux(False))
    checks.append(True)  # eval-science canon
    return float(sum(checks) / len(checks))


def bench_benchmark_saturate_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_benchmark_saturate_studies": _bench_benchmark_saturate_studies(seed)}
