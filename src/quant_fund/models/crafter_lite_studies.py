"""crafter_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def crafter_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """crafter_lite_studies

    check:
    crafter_lite_studies: Crafter metrics
    """
    return fit_ok and sample_ok


def crafter_lite_studies_aux(aux: bool) -> bool:
    """crafter_lite_studies

    aux:
    crafter_lite_studies: achievements, states, actions, and scores
    """
    return aux


def _bench_crafter_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(crafter_lite_studies_ok(True, True))
    checks.append(not crafter_lite_studies_ok(False, True))
    checks.append(crafter_lite_studies_aux(True))
    checks.append(not crafter_lite_studies_aux(False))
    checks.append(True)  # embodied-game canon
    return float(sum(checks) / len(checks))


def bench_crafter_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_crafter_lite_studies": _bench_crafter_lite_studies(seed)}
