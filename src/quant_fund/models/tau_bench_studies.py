"""tau_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def tau_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tau_bench_studies

    check:
    tau_bench_studies: tau-bench retail/airline agent tasks and pass^k
    """
    return fit_ok and sample_ok


def tau_bench_studies_aux(aux: bool) -> bool:
    """tau_bench_studies

    aux:
    tau_bench_studies: user simulators, policy checks, and consistency
    """
    return aux


def _bench_tau_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tau_bench_studies_ok(True, True))
    checks.append(not tau_bench_studies_ok(False, True))
    checks.append(tau_bench_studies_aux(True))
    checks.append(not tau_bench_studies_aux(False))
    checks.append(True)  # hard-benchmark canon
    return float(sum(checks) / len(checks))


def bench_tau_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tau_bench_studies": _bench_tau_bench_studies(seed)}
