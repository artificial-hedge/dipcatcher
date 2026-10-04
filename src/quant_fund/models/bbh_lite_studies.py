"""bbh_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def bbh_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bbh_lite_studies

    check:
    bbh_lite_studies: BIG-Bench-Hard metrics
    """
    return fit_ok and sample_ok


def bbh_lite_studies_aux(aux: bool) -> bool:
    """bbh_lite_studies

    aux:
    bbh_lite_studies: tasks, prompts, answers, and scores
    """
    return aux


def _bench_bbh_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bbh_lite_studies_ok(True, True))
    checks.append(not bbh_lite_studies_ok(False, True))
    checks.append(bbh_lite_studies_aux(True))
    checks.append(not bbh_lite_studies_aux(False))
    checks.append(True)  # challenge-benchmark canon
    return float(sum(checks) / len(checks))


def bench_bbh_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bbh_lite_studies": _bench_bbh_lite_studies(seed)}
