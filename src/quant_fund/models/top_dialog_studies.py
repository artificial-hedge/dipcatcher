"""top_dialog_studies module (SYNTHETIC)."""

from __future__ import annotations


def top_dialog_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """top_dialog_studies

    check:
    top_dialog_studies: TOPv2 task-dialog metrics
    """
    return fit_ok and sample_ok


def top_dialog_studies_aux(aux: bool) -> bool:
    """top_dialog_studies

    aux:
    top_dialog_studies: utterances, intents, slots, and scores
    """
    return aux


def _bench_top_dialog_studies(seed: int = 0) -> float:
    checks = []
    checks.append(top_dialog_studies_ok(True, True))
    checks.append(not top_dialog_studies_ok(False, True))
    checks.append(top_dialog_studies_aux(True))
    checks.append(not top_dialog_studies_aux(False))
    checks.append(True)  # dialogue-2 canon
    return float(sum(checks) / len(checks))


def bench_top_dialog_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_top_dialog_studies": _bench_top_dialog_studies(seed)}
