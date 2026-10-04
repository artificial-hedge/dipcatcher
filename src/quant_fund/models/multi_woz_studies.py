"""multi_woz_studies module (SYNTHETIC)."""

from __future__ import annotations


def multi_woz_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """multi_woz_studies

    check:
    multi_woz_studies: MultiWOZ metrics
    """
    return fit_ok and sample_ok


def multi_woz_studies_aux(aux: bool) -> bool:
    """multi_woz_studies

    aux:
    multi_woz_studies: domains, acts, slots, and scores
    """
    return aux


def _bench_multi_woz_studies(seed: int = 0) -> float:
    checks = []
    checks.append(multi_woz_studies_ok(True, True))
    checks.append(not multi_woz_studies_ok(False, True))
    checks.append(multi_woz_studies_aux(True))
    checks.append(not multi_woz_studies_aux(False))
    checks.append(True)  # dialogue-2 canon
    return float(sum(checks) / len(checks))


def bench_multi_woz_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_multi_woz_studies": _bench_multi_woz_studies(seed)}
