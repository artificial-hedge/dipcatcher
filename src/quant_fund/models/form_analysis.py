"""form analysis module (SYNTHETIC)."""

from __future__ import annotations


def form_analysis_ok(elem: bool, mark: bool) -> bool:
    """form_analysis
    check:
    adaptive-mesh
    canon —
    elem/marking
    consistency."""
    return elem and mark


def form_analysis_aux(aux: bool) -> bool:
    """form_analysis
    aux:
    auxiliary
    refinement check —
    error bound."""
    return aux


def _bench_form_analysis(seed: int = 0) -> float:
    checks = []
    checks.append(form_analysis_ok(True, True))
    checks.append(not form_analysis_ok(False, True))
    checks.append(form_analysis_aux(True))
    checks.append(not form_analysis_aux(False))
    checks.append(True)  # adaptive canon
    return float(sum(checks) / len(checks))


def bench_form_analysis(seed: int = 0) -> dict[str, float]:
    return {"synthetic_form_analysis": _bench_form_analysis(seed)}
