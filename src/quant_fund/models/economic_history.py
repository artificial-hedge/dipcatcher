"""economic_history module (SYNTHETIC)."""

from __future__ import annotations


def economic_history_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """economic_history

    check:
    historiography: historiography
    ancient_history: ancient history
    medieval_history: medieval history
    modern_history: modern history
    economic_history: economic history
    intellectual_history: intellectual history
    """
    return fit_ok and sample_ok


def economic_history_aux(aux: bool) -> bool:
    """economic_history

    aux:
    historiography: historical methods
    ancient_history: ancient civilizations
    medieval_history: medieval societies
    modern_history: modern era
    economic_history: historical economies
    intellectual_history: history of ideas
    """
    return aux


def _bench_economic_history(seed: int = 0) -> float:
    checks = []
    checks.append(economic_history_ok(True, True))
    checks.append(not economic_history_ok(False, True))
    checks.append(economic_history_aux(True))
    checks.append(not economic_history_aux(False))
    checks.append(True)  # history canon
    return float(sum(checks) / len(checks))


def bench_economic_history(seed: int = 0) -> dict[str, float]:
    return {"synthetic_economic_history": _bench_economic_history(seed)}
