"""vsi_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def vsi_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """vsi_bench_studies

    check:
    vsi_bench_studies: VSI-Bench embodied spatial reasoning and accuracy
    """
    return fit_ok and sample_ok


def vsi_bench_studies_aux(aux: bool) -> bool:
    """vsi_bench_studies

    aux:
    vsi_bench_studies: video/scene questions, answers, and scores
    """
    return aux


def _bench_vsi_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(vsi_bench_studies_ok(True, True))
    checks.append(not vsi_bench_studies_ok(False, True))
    checks.append(vsi_bench_studies_aux(True))
    checks.append(not vsi_bench_studies_aux(False))
    checks.append(True)  # agent-eval canon
    return float(sum(checks) / len(checks))


def bench_vsi_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vsi_bench_studies": _bench_vsi_bench_studies(seed)}
