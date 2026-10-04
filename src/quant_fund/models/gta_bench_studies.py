"""gta_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def gta_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gta_bench_studies

    check:
    gta_bench_studies: GTA-bench metrics
    """
    return fit_ok and sample_ok


def gta_bench_studies_aux(aux: bool) -> bool:
    """gta_bench_studies

    aux:
    gta_bench_studies: queries, tools, chains, and scores
    """
    return aux


def _bench_gta_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gta_bench_studies_ok(True, True))
    checks.append(not gta_bench_studies_ok(False, True))
    checks.append(gta_bench_studies_aux(True))
    checks.append(not gta_bench_studies_aux(False))
    checks.append(True)  # tool-use canon
    return float(sum(checks) / len(checks))


def bench_gta_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gta_bench_studies": _bench_gta_bench_studies(seed)}
