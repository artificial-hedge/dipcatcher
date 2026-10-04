"""apps_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def apps_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """apps_bench_studies

    check:
    apps_bench_studies: APPS competitive-programming solve metrics
    """
    return fit_ok and sample_ok


def apps_bench_studies_aux(aux: bool) -> bool:
    """apps_bench_studies

    aux:
    apps_bench_studies: problems, levels, and pass rates
    """
    return aux


def _bench_apps_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(apps_bench_studies_ok(True, True))
    checks.append(not apps_bench_studies_ok(False, True))
    checks.append(apps_bench_studies_aux(True))
    checks.append(not apps_bench_studies_aux(False))
    checks.append(True)  # code-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_apps_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_apps_bench_studies": _bench_apps_bench_studies(seed)}
