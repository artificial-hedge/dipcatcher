"""interventional_radiology module (SYNTHETIC)."""

from __future__ import annotations


def interventional_radiology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """interventional_radiology

    check:
    dialysis_technology: dialysis technology
    transplant_studies: transplant studies
    hepatobiliary_studies: hepatobiliary studies
    cardiac_electrophysiology: cardiac electrophysiology
    interventional_radiology: interventional radiology
    nuclear_cardiology: nuclear cardiology
    """
    return fit_ok and sample_ok


def interventional_radiology_aux(aux: bool) -> bool:
    """interventional_radiology

    aux:
    dialysis_technology: membranes and ultrafiltration
    transplant_studies: grafts and immunosuppression
    hepatobiliary_studies: bile ducts and liver
    cardiac_electrophysiology: ablations and mappings
    interventional_radiology: catheters and embolization
    nuclear_cardiology: tracers and perfusion
    """
    return aux


def _bench_interventional_radiology(seed: int = 0) -> float:
    checks = []
    checks.append(interventional_radiology_ok(True, True))
    checks.append(not interventional_radiology_ok(False, True))
    checks.append(interventional_radiology_aux(True))
    checks.append(not interventional_radiology_aux(False))
    checks.append(True)  # interventional-medicine canon
    return float(sum(checks) / len(checks))


def bench_interventional_radiology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_interventional_radiology": _bench_interventional_radiology(seed)}
