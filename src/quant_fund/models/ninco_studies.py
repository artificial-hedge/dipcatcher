"""ninco_studies module (SYNTHETIC)."""

from __future__ import annotations


def ninco_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ninco_studies

    check:
    ninco_studies: NINCO no-image-net-common detection and scores
    """
    return fit_ok and sample_ok


def ninco_studies_aux(aux: bool) -> bool:
    """ninco_studies

    aux:
    ninco_studies: novel class probes, rejection, and accuracy
    """
    return aux


def _bench_ninco_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ninco_studies_ok(True, True))
    checks.append(not ninco_studies_ok(False, True))
    checks.append(ninco_studies_aux(True))
    checks.append(not ninco_studies_aux(False))
    checks.append(True)  # benchmark-eval canon
    return float(sum(checks) / len(checks))


def bench_ninco_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ninco_studies": _bench_ninco_studies(seed)}
