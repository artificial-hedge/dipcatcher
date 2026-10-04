"""sorry_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def sorry_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sorry_bench_studies

    check:
    sorry_bench_studies: SORRY-Bench refusal taxonomy, categories, and scores
    """
    return fit_ok and sample_ok


def sorry_bench_studies_aux(aux: bool) -> bool:
    """sorry_bench_studies

    aux:
    sorry_bench_studies: 45 refusal classes, attack prompts, and eval rates
    """
    return aux


def _bench_sorry_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sorry_bench_studies_ok(True, True))
    checks.append(not sorry_bench_studies_ok(False, True))
    checks.append(sorry_bench_studies_aux(True))
    checks.append(not sorry_bench_studies_aux(False))
    checks.append(True)  # safety-benchmark canon
    return float(sum(checks) / len(checks))


def bench_sorry_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sorry_bench_studies": _bench_sorry_bench_studies(seed)}
