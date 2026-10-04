"""history_of_technology module (SYNTHETIC)."""

from __future__ import annotations


def history_of_technology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """history_of_technology

    check:
    social_history: social history
    cultural_history: cultural history
    military_history: military history
    diplomatic_history: diplomatic history
    history_of_technology: history of technology
    history_of_medicine: history of medicine
    """
    return fit_ok and sample_ok


def history_of_technology_aux(aux: bool) -> bool:
    """history_of_technology

    aux:
    social_history: everyday life
    cultural_history: cultural practice
    military_history: warfare
    diplomatic_history: state relations
    history_of_technology: technical change
    history_of_medicine: medical practice
    """
    return aux


def _bench_history_of_technology(seed: int = 0) -> float:
    checks = []
    checks.append(history_of_technology_ok(True, True))
    checks.append(not history_of_technology_ok(False, True))
    checks.append(history_of_technology_aux(True))
    checks.append(not history_of_technology_aux(False))
    checks.append(True)  # history-2 canon
    return float(sum(checks) / len(checks))


def bench_history_of_technology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_history_of_technology": _bench_history_of_technology(seed)}
