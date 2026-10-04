"""infinite_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def infinite_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """infinite_bench_studies

    check:
    infinite_bench_studies: InfiniteBench long-doc QA and retrieval metrics
    """
    return fit_ok and sample_ok


def infinite_bench_studies_aux(aux: bool) -> bool:
    """infinite_bench_studies

    aux:
    infinite_bench_studies: passkey, code-run, math-find, and novel-QA tasks
    """
    return aux


def _bench_infinite_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(infinite_bench_studies_ok(True, True))
    checks.append(not infinite_bench_studies_ok(False, True))
    checks.append(infinite_bench_studies_aux(True))
    checks.append(not infinite_bench_studies_aux(False))
    checks.append(True)  # long-context-factuality canon
    return float(sum(checks) / len(checks))


def bench_infinite_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_infinite_bench_studies": _bench_infinite_bench_studies(seed)}
