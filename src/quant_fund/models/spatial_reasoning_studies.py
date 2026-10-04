"""spatial_reasoning_studies module (SYNTHETIC)."""

from __future__ import annotations


def spatial_reasoning_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """spatial_reasoning_studies

    check:
    spatial_reasoning_studies: 3D relations and viewpoint transforms/poses and frames
    """
    return fit_ok and sample_ok


def spatial_reasoning_studies_aux(aux: bool) -> bool:
    """spatial_reasoning_studies

    aux:
    spatial_reasoning_studies: spatial chain-of-thought and grounding/objects and layouts
    """
    return aux


def _bench_spatial_reasoning_studies(seed: int = 0) -> float:
    checks = []
    checks.append(spatial_reasoning_studies_ok(True, True))
    checks.append(not spatial_reasoning_studies_ok(False, True))
    checks.append(spatial_reasoning_studies_aux(True))
    checks.append(not spatial_reasoning_studies_aux(False))
    checks.append(True)  # embodied-VLA canon
    return float(sum(checks) / len(checks))


def bench_spatial_reasoning_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spatial_reasoning_studies": _bench_spatial_reasoning_studies(seed)}
