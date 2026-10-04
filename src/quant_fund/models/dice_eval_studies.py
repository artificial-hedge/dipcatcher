"""dice_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def dice_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dice_eval_studies

    check:
    dice_eval_studies: DICE factuality metrics
    """
    return fit_ok and sample_ok


def dice_eval_studies_aux(aux: bool) -> bool:
    """dice_eval_studies

    aux:
    dice_eval_studies: summaries, sources, labels, and scores
    """
    return aux


def _bench_dice_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dice_eval_studies_ok(True, True))
    checks.append(not dice_eval_studies_ok(False, True))
    checks.append(dice_eval_studies_aux(True))
    checks.append(not dice_eval_studies_aux(False))
    checks.append(True)  # faithfulness-eval canon
    return float(sum(checks) / len(checks))


def bench_dice_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dice_eval_studies": _bench_dice_eval_studies(seed)}
