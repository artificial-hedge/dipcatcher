"""alpaca_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def alpaca_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """alpaca_eval_studies

    check:
    alpaca_eval_studies: AlpacaEval instruction-following win rates
    """
    return fit_ok and sample_ok


def alpaca_eval_studies_aux(aux: bool) -> bool:
    """alpaca_eval_studies

    aux:
    alpaca_eval_studies: instructions, model outputs, and judge prefs
    """
    return aux


def _bench_alpaca_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(alpaca_eval_studies_ok(True, True))
    checks.append(not alpaca_eval_studies_ok(False, True))
    checks.append(alpaca_eval_studies_aux(True))
    checks.append(not alpaca_eval_studies_aux(False))
    checks.append(True)  # generation-quality canon
    return float(sum(checks) / len(checks))


def bench_alpaca_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_alpaca_eval_studies": _bench_alpaca_eval_studies(seed)}
