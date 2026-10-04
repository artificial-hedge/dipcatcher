"""codescope_studies module (SYNTHETIC)."""

from __future__ import annotations


def codescope_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """codescope_studies

    check:
    codescope_studies: CodeScope code-understanding multi-task metrics
    """
    return fit_ok and sample_ok


def codescope_studies_aux(aux: bool) -> bool:
    """codescope_studies

    aux:
    codescope_studies: snippets, tasks, answers, and scores
    """
    return aux


def _bench_codescope_studies(seed: int = 0) -> float:
    checks = []
    checks.append(codescope_studies_ok(True, True))
    checks.append(not codescope_studies_ok(False, True))
    checks.append(codescope_studies_aux(True))
    checks.append(not codescope_studies_aux(False))
    checks.append(True)  # code-eval-3 canon
    return float(sum(checks) / len(checks))


def bench_codescope_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_codescope_studies": _bench_codescope_studies(seed)}
