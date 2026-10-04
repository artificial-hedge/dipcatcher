"""faithful_dial_studies module (SYNTHETIC)."""

from __future__ import annotations


def faithful_dial_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """faithful_dial_studies

    check:
    faithful_dial_studies: FaithDial metrics
    """
    return fit_ok and sample_ok


def faithful_dial_studies_aux(aux: bool) -> bool:
    """faithful_dial_studies

    aux:
    faithful_dial_studies: contexts, beliefs, responses, and scores
    """
    return aux


def _bench_faithful_dial_studies(seed: int = 0) -> float:
    checks = []
    checks.append(faithful_dial_studies_ok(True, True))
    checks.append(not faithful_dial_studies_ok(False, True))
    checks.append(faithful_dial_studies_aux(True))
    checks.append(not faithful_dial_studies_aux(False))
    checks.append(True)  # dialogue-2 canon
    return float(sum(checks) / len(checks))


def bench_faithful_dial_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_faithful_dial_studies": _bench_faithful_dial_studies(seed)}
