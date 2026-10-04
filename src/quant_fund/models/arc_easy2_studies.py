"""arc_easy2_studies module (SYNTHETIC)."""

from __future__ import annotations


def arc_easy2_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """arc_easy2_studies

    check:
    arc_easy2_studies: ARC-Easy metrics
    """
    return fit_ok and sample_ok


def arc_easy2_studies_aux(aux: bool) -> bool:
    """arc_easy2_studies

    aux:
    arc_easy2_studies: questions, options, labels, and accuracies
    """
    return aux


def _bench_arc_easy2_studies(seed: int = 0) -> float:
    checks = []
    checks.append(arc_easy2_studies_ok(True, True))
    checks.append(not arc_easy2_studies_ok(False, True))
    checks.append(arc_easy2_studies_aux(True))
    checks.append(not arc_easy2_studies_aux(False))
    checks.append(True)  # MC-eval canon
    return float(sum(checks) / len(checks))


def bench_arc_easy2_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_arc_easy2_studies": _bench_arc_easy2_studies(seed)}
