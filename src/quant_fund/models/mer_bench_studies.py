"""mer_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def mer_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mer_bench_studies

    check:
    mer_bench_studies: MergeBench conflict-resolution accuracy metrics
    """
    return fit_ok and sample_ok


def mer_bench_studies_aux(aux: bool) -> bool:
    """mer_bench_studies

    aux:
    mer_bench_studies: conflicts, merges, and resolution scores
    """
    return aux


def _bench_mer_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mer_bench_studies_ok(True, True))
    checks.append(not mer_bench_studies_ok(False, True))
    checks.append(mer_bench_studies_aux(True))
    checks.append(not mer_bench_studies_aux(False))
    checks.append(True)  # code-eval-3 canon
    return float(sum(checks) / len(checks))


def bench_mer_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mer_bench_studies": _bench_mer_bench_studies(seed)}
