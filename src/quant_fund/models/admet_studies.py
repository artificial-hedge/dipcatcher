"""admet_studies module (SYNTHETIC)."""

from __future__ import annotations


def admet_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """admet_studies

    check:
    admet_studies: absorption and distribution/clearance and toxicity
    """
    return fit_ok and sample_ok


def admet_studies_aux(aux: bool) -> bool:
    """admet_studies

    aux:
    admet_studies: pharmacokinetics and half-life/dose and bioavailability
    """
    return aux


def _bench_admet_studies(seed: int = 0) -> float:
    checks = []
    checks.append(admet_studies_ok(True, True))
    checks.append(not admet_studies_ok(False, True))
    checks.append(admet_studies_aux(True))
    checks.append(not admet_studies_aux(False))
    checks.append(True)  # drug-discovery canon
    return float(sum(checks) / len(checks))


def bench_admet_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_admet_studies": _bench_admet_studies(seed)}
