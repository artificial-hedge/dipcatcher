"""monosemantic_studies module (SYNTHETIC)."""

from __future__ import annotations


def monosemantic_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """monosemantic_studies

    check:
    monosemantic_studies: dictionary features and disentanglement/purity and coverage
    """
    return fit_ok and sample_ok


def monosemantic_studies_aux(aux: bool) -> bool:
    """monosemantic_studies

    aux:
    monosemantic_studies: feature dashboards and automated explanations/interpretability and scale
    """
    return aux


def _bench_monosemantic_studies(seed: int = 0) -> float:
    checks = []
    checks.append(monosemantic_studies_ok(True, True))
    checks.append(not monosemantic_studies_ok(False, True))
    checks.append(monosemantic_studies_aux(True))
    checks.append(not monosemantic_studies_aux(False))
    checks.append(True)  # interpretability-3 canon
    return float(sum(checks) / len(checks))


def bench_monosemantic_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_monosemantic_studies": _bench_monosemantic_studies(seed)}
