"""api_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def api_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """api_eval_studies

    check:
    api_eval_studies: API-Eval metrics
    """
    return fit_ok and sample_ok


def api_eval_studies_aux(aux: bool) -> bool:
    """api_eval_studies

    aux:
    api_eval_studies: specs, calls, responses, and scores
    """
    return aux


def _bench_api_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(api_eval_studies_ok(True, True))
    checks.append(not api_eval_studies_ok(False, True))
    checks.append(api_eval_studies_aux(True))
    checks.append(not api_eval_studies_aux(False))
    checks.append(True)  # code-agent canon
    return float(sum(checks) / len(checks))


def bench_api_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_api_eval_studies": _bench_api_eval_studies(seed)}
