"""safety_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def safety_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """safety_bench_studies

    check:
    safety_bench_studies: safety-eval categories/items and safe rates
    """
    return fit_ok and sample_ok


def safety_bench_studies_aux(aux: bool) -> bool:
    """safety_bench_studies

    aux:
    safety_bench_studies: hazard taxonomy prompts/judges and labels
    """
    return aux


def _bench_safety_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(safety_bench_studies_ok(True, True))
    checks.append(not safety_bench_studies_ok(False, True))
    checks.append(safety_bench_studies_aux(True))
    checks.append(not safety_bench_studies_aux(False))
    checks.append(True)  # safety-eval canon
    return float(sum(checks) / len(checks))


def bench_safety_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_safety_bench_studies": _bench_safety_bench_studies(seed)}
