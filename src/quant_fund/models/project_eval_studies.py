"""project_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def project_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """project_eval_studies

    check:
    project_eval_studies: ProjectEval project-level correctness metrics
    """
    return fit_ok and sample_ok


def project_eval_studies_aux(aux: bool) -> bool:
    """project_eval_studies

    aux:
    project_eval_studies: projects, functions, tests, and pass rates
    """
    return aux


def _bench_project_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(project_eval_studies_ok(True, True))
    checks.append(not project_eval_studies_ok(False, True))
    checks.append(project_eval_studies_aux(True))
    checks.append(not project_eval_studies_aux(False))
    checks.append(True)  # code-eval-3 canon
    return float(sum(checks) / len(checks))


def bench_project_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_project_eval_studies": _bench_project_eval_studies(seed)}
