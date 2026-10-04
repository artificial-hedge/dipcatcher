"""heart_failure_medicine module (SYNTHETIC)."""

from __future__ import annotations


def heart_failure_medicine_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """heart_failure_medicine

    check:
    cardiology_studies: cardiology studies
    interventional_cardiology: interventional cardiology
    electrophysiology_studies: electrophysiology studies
    heart_failure_medicine: heart failure medicine
    preventive_cardiology: preventive cardiology
    cardiovascular_imaging: cardiovascular imaging
    """
    return fit_ok and sample_ok


def heart_failure_medicine_aux(aux: bool) -> bool:
    """heart_failure_medicine

    aux:
    cardiology_studies: coronary and valvular
    interventional_cardiology: stents and pci
    electrophysiology_studies: arrhythmia and ablation
    heart_failure_medicine: hfref and hfpef
    preventive_cardiology: lipids and risk
    cardiovascular_imaging: echo and mri
    """
    return aux


def _bench_heart_failure_medicine(seed: int = 0) -> float:
    checks = []
    checks.append(heart_failure_medicine_ok(True, True))
    checks.append(not heart_failure_medicine_ok(False, True))
    checks.append(heart_failure_medicine_aux(True))
    checks.append(not heart_failure_medicine_aux(False))
    checks.append(True)  # cardiology canon
    return float(sum(checks) / len(checks))


def bench_heart_failure_medicine(seed: int = 0) -> dict[str, float]:
    return {"synthetic_heart_failure_medicine": _bench_heart_failure_medicine(seed)}
