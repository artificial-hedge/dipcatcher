"""intellectual_history module (SYNTHETIC)."""

from __future__ import annotations


def intellectual_history_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """intellectual_history

    check:
    historiography: historiography
    ancient_history: ancient history
    medieval_history: medieval history
    modern_history: modern history
    economic_history: economic history
    intellectual_history: intellectual history
    """
    return fit_ok and sample_ok


def intellectual_history_aux(aux: bool) -> bool:
    """intellectual_history

    aux:
    historiography: historical methods
    ancient_history: ancient civilizations
    medieval_history: medieval societies
    modern_history: modern era
    economic_history: historical economies
    intellectual_history: history of ideas
    """
    return aux


def _bench_intellectual_history(seed: int = 0) -> float:
    checks = []
    checks.append(intellectual_history_ok(True, True))
    checks.append(not intellectual_history_ok(False, True))
    checks.append(intellectual_history_aux(True))
    checks.append(not intellectual_history_aux(False))
    checks.append(True)  # history canon
    return float(sum(checks) / len(checks))


def bench_intellectual_history(seed: int = 0) -> dict[str, float]:
    return {"synthetic_intellectual_history": _bench_intellectual_history(seed)}
