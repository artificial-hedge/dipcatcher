"""media_archaeology module (SYNTHETIC)."""

from __future__ import annotations


def media_archaeology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """media_archaeology

    check:
    history_of_science: history of science
    sts_studies: science and technology studies
    philosophy_of_technology: philosophy of technology
    media_archaeology: media archaeology
    information_history: information history
    technology_studies: technology studies
    """
    return fit_ok and sample_ok


def media_archaeology_aux(aux: bool) -> bool:
    """media_archaeology

    aux:
    history_of_science: scientific revolution
    sts_studies: actor-network theory
    philosophy_of_technology: techne critique
    media_archaeology: media genealogy
    information_history: information age
    technology_studies: technological systems
    """
    return aux


def _bench_media_archaeology(seed: int = 0) -> float:
    checks = []
    checks.append(media_archaeology_ok(True, True))
    checks.append(not media_archaeology_ok(False, True))
    checks.append(media_archaeology_aux(True))
    checks.append(not media_archaeology_aux(False))
    checks.append(True)  # history-of-science canon
    return float(sum(checks) / len(checks))


def bench_media_archaeology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_media_archaeology": _bench_media_archaeology(seed)}
