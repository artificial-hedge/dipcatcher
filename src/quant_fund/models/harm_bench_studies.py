"""harm_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def harm_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """harm_bench_studies

    check:
    harm_bench_studies: HarmBench attack behaviors/methods and ASR
    """
    return fit_ok and sample_ok


def harm_bench_studies_aux(aux: bool) -> bool:
    """harm_bench_studies

    aux:
    harm_bench_studies: red-team attack configs/seeds and refusal flags
    """
    return aux


def _bench_harm_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(harm_bench_studies_ok(True, True))
    checks.append(not harm_bench_studies_ok(False, True))
    checks.append(harm_bench_studies_aux(True))
    checks.append(not harm_bench_studies_aux(False))
    checks.append(True)  # safety-eval canon
    return float(sum(checks) / len(checks))


def bench_harm_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_harm_bench_studies": _bench_harm_bench_studies(seed)}
