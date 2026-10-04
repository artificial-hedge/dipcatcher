"""visual_grounding_studies module (SYNTHETIC)."""

from __future__ import annotations


def visual_grounding_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """visual_grounding_studies

    check:
    visual_grounding_studies: referring expressions and bounding boxes/pointing and grounding
    """
    return fit_ok and sample_ok


def visual_grounding_studies_aux(aux: bool) -> bool:
    """visual_grounding_studies

    aux:
    visual_grounding_studies: detection alignment and regions/coordinates and comprehension
    """
    return aux


def _bench_visual_grounding_studies(seed: int = 0) -> float:
    checks = []
    checks.append(visual_grounding_studies_ok(True, True))
    checks.append(not visual_grounding_studies_ok(False, True))
    checks.append(visual_grounding_studies_aux(True))
    checks.append(not visual_grounding_studies_aux(False))
    checks.append(True)  # omni-modal canon
    return float(sum(checks) / len(checks))


def bench_visual_grounding_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_visual_grounding_studies": _bench_visual_grounding_studies(seed)}
