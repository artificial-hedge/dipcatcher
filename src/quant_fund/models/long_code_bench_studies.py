"""long_code_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def long_code_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """long_code_bench_studies

    check:
    long_code_bench_studies: Long-horizon code-generation metrics
    """
    return fit_ok and sample_ok


def long_code_bench_studies_aux(aux: bool) -> bool:
    """long_code_bench_studies

    aux:
    long_code_bench_studies: instructions, contexts, edits, and scores
    """
    return aux


def _bench_long_code_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(long_code_bench_studies_ok(True, True))
    checks.append(not long_code_bench_studies_ok(False, True))
    checks.append(long_code_bench_studies_aux(True))
    checks.append(not long_code_bench_studies_aux(False))
    checks.append(True)  # code-eval-4 canon
    return float(sum(checks) / len(checks))


def bench_long_code_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_long_code_bench_studies": _bench_long_code_bench_studies(seed)}
