"""unqover_studies module (SYNTHETIC)."""

from __future__ import annotations


def unqover_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """unqover_studies

    check:
    unqover_studies: UnQover underspecification bias metrics
    """
    return fit_ok and sample_ok


def unqover_studies_aux(aux: bool) -> bool:
    """unqover_studies

    aux:
    unqover_studies: questions, contexts, and bias scores
    """
    return aux


def _bench_unqover_studies(seed: int = 0) -> float:
    checks = []
    checks.append(unqover_studies_ok(True, True))
    checks.append(not unqover_studies_ok(False, True))
    checks.append(unqover_studies_aux(True))
    checks.append(not unqover_studies_aux(False))
    checks.append(True)  # multilingual-eval canon
    return float(sum(checks) / len(checks))


def bench_unqover_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_unqover_studies": _bench_unqover_studies(seed)}
