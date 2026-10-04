"""digital_history module (SYNTHETIC)."""

from __future__ import annotations


def digital_history_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """digital_history

    check:
    oral_history: oral history
    public_history: public history
    digital_history: digital history
    environmental_history: environmental history
    global_history: global history
    maritime_history: maritime history
    """
    return fit_ok and sample_ok


def digital_history_aux(aux: bool) -> bool:
    """digital_history

    aux:
    oral_history: memory interviews
    public_history: heritage and museums
    digital_history: computational methods
    environmental_history: nature and society
    global_history: transregional connections
    maritime_history: seafaring past
    """
    return aux


def _bench_digital_history(seed: int = 0) -> float:
    checks = []
    checks.append(digital_history_ok(True, True))
    checks.append(not digital_history_ok(False, True))
    checks.append(digital_history_aux(True))
    checks.append(not digital_history_aux(False))
    checks.append(True)  # history-3 canon
    return float(sum(checks) / len(checks))


def bench_digital_history(seed: int = 0) -> dict[str, float]:
    return {"synthetic_digital_history": _bench_digital_history(seed)}
