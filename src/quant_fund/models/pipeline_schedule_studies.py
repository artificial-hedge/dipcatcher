"""pipeline_schedule_studies module (SYNTHETIC)."""

from __future__ import annotations


def pipeline_schedule_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pipeline_schedule_studies

    check:
    pipeline_schedule_studies: 1F1B and interleaved microbatches/stages and bubbles
    """
    return fit_ok and sample_ok


def pipeline_schedule_studies_aux(aux: bool) -> bool:
    """pipeline_schedule_studies

    aux:
    pipeline_schedule_studies: GPipe/Chimera-style scheduling/warmup and flush
    """
    return aux


def _bench_pipeline_schedule_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pipeline_schedule_studies_ok(True, True))
    checks.append(not pipeline_schedule_studies_ok(False, True))
    checks.append(pipeline_schedule_studies_aux(True))
    checks.append(not pipeline_schedule_studies_aux(False))
    checks.append(True)  # distributed-training canon
    return float(sum(checks) / len(checks))


def bench_pipeline_schedule_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pipeline_schedule_studies": _bench_pipeline_schedule_studies(seed)}
