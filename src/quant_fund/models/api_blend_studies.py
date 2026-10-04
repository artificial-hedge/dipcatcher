"""api_blend_studies module (SYNTHETIC)."""

from __future__ import annotations


def api_blend_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """api_blend_studies

    check:
    api_blend_studies: API-Blend metrics
    """
    return fit_ok and sample_ok


def api_blend_studies_aux(aux: bool) -> bool:
    """api_blend_studies

    aux:
    api_blend_studies: tasks, calls, sequences, and scores
    """
    return aux


def _bench_api_blend_studies(seed: int = 0) -> float:
    checks = []
    checks.append(api_blend_studies_ok(True, True))
    checks.append(not api_blend_studies_ok(False, True))
    checks.append(api_blend_studies_aux(True))
    checks.append(not api_blend_studies_aux(False))
    checks.append(True)  # tool-use canon
    return float(sum(checks) / len(checks))


def bench_api_blend_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_api_blend_studies": _bench_api_blend_studies(seed)}
