"""screen_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def screen_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """screen_eval_studies

    check:
    screen_eval_studies: ScreenEval UI-grounding action accuracy and metrics
    """
    return fit_ok and sample_ok


def screen_eval_studies_aux(aux: bool) -> bool:
    """screen_eval_studies

    aux:
    screen_eval_studies: screens, instructions, targets, and hit rates
    """
    return aux


def _bench_screen_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(screen_eval_studies_ok(True, True))
    checks.append(not screen_eval_studies_ok(False, True))
    checks.append(screen_eval_studies_aux(True))
    checks.append(not screen_eval_studies_aux(False))
    checks.append(True)  # agent-eval canon
    return float(sum(checks) / len(checks))


def bench_screen_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_screen_eval_studies": _bench_screen_eval_studies(seed)}
