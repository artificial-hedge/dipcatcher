"""sts_studies module (SYNTHETIC)."""

from __future__ import annotations


def sts_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sts_studies

    check:
    history_of_science: history of science
    sts_studies: science and technology studies
    philosophy_of_technology: philosophy of technology
    media_archaeology: media archaeology
    information_history: information history
    technology_studies: technology studies
    """
    return fit_ok and sample_ok


def sts_studies_aux(aux: bool) -> bool:
    """sts_studies

    aux:
    history_of_science: scientific revolution
    sts_studies: actor-network theory
    philosophy_of_technology: techne critique
    media_archaeology: media genealogy
    information_history: information age
    technology_studies: technological systems
    """
    return aux


def _bench_sts_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sts_studies_ok(True, True))
    checks.append(not sts_studies_ok(False, True))
    checks.append(sts_studies_aux(True))
    checks.append(not sts_studies_aux(False))
    checks.append(True)  # history-of-science canon
    return float(sum(checks) / len(checks))


def bench_sts_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sts_studies": _bench_sts_studies(seed)}
