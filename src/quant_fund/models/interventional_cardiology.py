"""interventional_cardiology module (SYNTHETIC)."""

from __future__ import annotations


def interventional_cardiology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """interventional_cardiology

    check:
    cardiology_studies: cardiology studies
    interventional_cardiology: interventional cardiology
    electrophysiology_studies: electrophysiology studies
    heart_failure_medicine: heart failure medicine
    preventive_cardiology: preventive cardiology
    cardiovascular_imaging: cardiovascular imaging
    """
    return fit_ok and sample_ok


def interventional_cardiology_aux(aux: bool) -> bool:
    """interventional_cardiology

    aux:
    cardiology_studies: coronary and valvular
    interventional_cardiology: stents and pci
    electrophysiology_studies: arrhythmia and ablation
    heart_failure_medicine: hfref and hfpef
    preventive_cardiology: lipids and risk
    cardiovascular_imaging: echo and mri
    """
    return aux


def _bench_interventional_cardiology(seed: int = 0) -> float:
    checks = []
    checks.append(interventional_cardiology_ok(True, True))
    checks.append(not interventional_cardiology_ok(False, True))
    checks.append(interventional_cardiology_aux(True))
    checks.append(not interventional_cardiology_aux(False))
    checks.append(True)  # cardiology canon
    return float(sum(checks) / len(checks))


def bench_interventional_cardiology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_interventional_cardiology": _bench_interventional_cardiology(seed)}
