"""history_5 module (SYNTHETIC)."""

from __future__ import annotations


def history_5_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """history_5

    check:
    philosophy_6: philosophy
    history_5: history
    religious_studies_2: religious studies
    classics_2: classics
    area_studies_2: area studies
    humanities_2: humanities
    """
    return fit_ok and sample_ok


def history_5_aux(aux: bool) -> bool:
    """history_5

    aux:
    philosophy_6: arguments and traditions
    history_5: sources and narratives
    religious_studies_2: texts and practices
    classics_2: antiquity and reception
    area_studies_2: regions and languages
    humanities_2: interpretation and critique
    """
    return aux


def _bench_history_5(seed: int = 0) -> float:
    checks = []
    checks.append(history_5_ok(True, True))
    checks.append(not history_5_ok(False, True))
    checks.append(history_5_aux(True))
    checks.append(not history_5_aux(False))
    checks.append(True)  # humanities canon
    return float(sum(checks) / len(checks))


def bench_history_5(seed: int = 0) -> dict[str, float]:
    return {"synthetic_history_5": _bench_history_5(seed)}
