"""thrombosis_medicine module (SYNTHETIC)."""

from __future__ import annotations


def thrombosis_medicine_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """thrombosis_medicine

    check:
    thrombosis_medicine: dvt and pe
    ..."""
    return fit_ok and sample_ok


def thrombosis_medicine_aux(aux: bool) -> bool:
    """thrombosis_medicine

    aux:
    thrombosis_medicine: anticoagulation and embolus
    ..."""
    return aux


def _bench_thrombosis_medicine(seed: int = 0) -> float:
    checks = []
    checks.append(thrombosis_medicine_ok(True, True))
    checks.append(not thrombosis_medicine_ok(False, True))
    checks.append(thrombosis_medicine_aux(True))
    checks.append(not thrombosis_medicine_aux(False))
    checks.append(True)  # hematology canon
    return float(sum(checks) / len(checks))


def bench_thrombosis_medicine(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thrombosis_medicine": _bench_thrombosis_medicine(seed)}
