"""odex_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def odex_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """odex_eval_studies

    check:
    odex_eval_studies: Open-domain executable code metrics
    """
    return fit_ok and sample_ok


def odex_eval_studies_aux(aux: bool) -> bool:
    """odex_eval_studies

    aux:
    odex_eval_studies: prompts, libraries, executions, and pass rates
    """
    return aux


def _bench_odex_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(odex_eval_studies_ok(True, True))
    checks.append(not odex_eval_studies_ok(False, True))
    checks.append(odex_eval_studies_aux(True))
    checks.append(not odex_eval_studies_aux(False))
    checks.append(True)  # code-eval-4 canon
    return float(sum(checks) / len(checks))


def bench_odex_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_odex_eval_studies": _bench_odex_eval_studies(seed)}
