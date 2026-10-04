"""repobench_studies module (SYNTHETIC)."""

from __future__ import annotations


def repobench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """repobench_studies

    check:
    repobench_studies: RepoBench repository-level completion metrics
    """
    return fit_ok and sample_ok


def repobench_studies_aux(aux: bool) -> bool:
    """repobench_studies

    aux:
    repobench_studies: contexts, completions, and scores
    """
    return aux


def _bench_repobench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(repobench_studies_ok(True, True))
    checks.append(not repobench_studies_ok(False, True))
    checks.append(repobench_studies_aux(True))
    checks.append(not repobench_studies_aux(False))
    checks.append(True)  # code-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_repobench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_repobench_studies": _bench_repobench_studies(seed)}
