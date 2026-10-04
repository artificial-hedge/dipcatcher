"""history_of_capitalism module (SYNTHETIC)."""

from __future__ import annotations


def history_of_capitalism_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """history_of_capitalism

    check:
    history_of_emotions: history of emotions
    history_of_sexuality: history of sexuality
    history_of_the_book: history of the book
    history_of_capitalism: history of capitalism
    history_of_religions: history of religions
    microhistory: microhistory
    """
    return fit_ok and sample_ok


def history_of_capitalism_aux(aux: bool) -> bool:
    """history_of_capitalism

    aux:
    history_of_emotions: feeling and society
    history_of_sexuality: desire and regulation
    history_of_the_book: print and readership
    history_of_capitalism: markets and labor
    history_of_religions: belief and practice
    microhistory: small-scale evidence
    """
    return aux


def _bench_history_of_capitalism(seed: int = 0) -> float:
    checks = []
    checks.append(history_of_capitalism_ok(True, True))
    checks.append(not history_of_capitalism_ok(False, True))
    checks.append(history_of_capitalism_aux(True))
    checks.append(not history_of_capitalism_aux(False))
    checks.append(True)  # history-4 canon
    return float(sum(checks) / len(checks))


def bench_history_of_capitalism(seed: int = 0) -> dict[str, float]:
    return {"synthetic_history_of_capitalism": _bench_history_of_capitalism(seed)}
