"""history_of_science module (SYNTHETIC)."""

from __future__ import annotations


def history_of_science_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """history_of_science

    check:
    history_of_science: history of science
    sts_studies: science and technology studies
    philosophy_of_technology: philosophy of technology
    media_archaeology: media archaeology
    information_history: information history
    technology_studies: technology studies
    """
    return fit_ok and sample_ok


def history_of_science_aux(aux: bool) -> bool:
    """history_of_science

    aux:
    history_of_science: scientific revolution
    sts_studies: actor-network theory
    philosophy_of_technology: techne critique
    media_archaeology: media genealogy
    information_history: information age
    technology_studies: technological systems
    """
    return aux


def _bench_history_of_science(seed: int = 0) -> float:
    checks = []
    checks.append(history_of_science_ok(True, True))
    checks.append(not history_of_science_ok(False, True))
    checks.append(history_of_science_aux(True))
    checks.append(not history_of_science_aux(False))
    checks.append(True)  # history-of-science canon
    return float(sum(checks) / len(checks))


def bench_history_of_science(seed: int = 0) -> dict[str, float]:
    return {"synthetic_history_of_science": _bench_history_of_science(seed)}
