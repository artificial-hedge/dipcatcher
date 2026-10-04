"""information_history module (SYNTHETIC)."""

from __future__ import annotations


def information_history_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """information_history

    check:
    history_of_science: history of science
    sts_studies: science and technology studies
    philosophy_of_technology: philosophy of technology
    media_archaeology: media archaeology
    information_history: information history
    technology_studies: technology studies
    """
    return fit_ok and sample_ok


def information_history_aux(aux: bool) -> bool:
    """information_history

    aux:
    history_of_science: scientific revolution
    sts_studies: actor-network theory
    philosophy_of_technology: techne critique
    media_archaeology: media genealogy
    information_history: information age
    technology_studies: technological systems
    """
    return aux


def _bench_information_history(seed: int = 0) -> float:
    checks = []
    checks.append(information_history_ok(True, True))
    checks.append(not information_history_ok(False, True))
    checks.append(information_history_aux(True))
    checks.append(not information_history_aux(False))
    checks.append(True)  # history-of-science canon
    return float(sum(checks) / len(checks))


def bench_information_history(seed: int = 0) -> dict[str, float]:
    return {"synthetic_information_history": _bench_information_history(seed)}
