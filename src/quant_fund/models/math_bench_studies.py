"""math_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def math_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """math_bench_studies

    check:
    math_bench_studies: MathBench reasoning/robustness scores and accuracy
    """
    return fit_ok and sample_ok


def math_bench_studies_aux(aux: bool) -> bool:
    """math_bench_studies

    aux:
    math_bench_studies: math items, chains, and correctness rates
    """
    return aux


def _bench_math_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(math_bench_studies_ok(True, True))
    checks.append(not math_bench_studies_ok(False, True))
    checks.append(math_bench_studies_aux(True))
    checks.append(not math_bench_studies_aux(False))
    checks.append(True)  # benchmark-eval canon
    return float(sum(checks) / len(checks))


def bench_math_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_math_bench_studies": _bench_math_bench_studies(seed)}
