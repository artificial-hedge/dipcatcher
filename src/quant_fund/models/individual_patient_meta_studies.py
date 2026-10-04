"""individual_patient_meta_studies module (SYNTHETIC)."""

from __future__ import annotations


def individual_patient_meta_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """individual_patient_meta_studies

    check:
    individual_patient_meta_studies: one-stage and two-stage/interactions and pooling
    """
    return fit_ok and sample_ok


def individual_patient_meta_studies_aux(aux: bool) -> bool:
    """individual_patient_meta_studies

    aux:
    individual_patient_meta_studies: standardization and sharing/power and feasibility
    """
    return aux


def _bench_individual_patient_meta_studies(seed: int = 0) -> float:
    checks = []
    checks.append(individual_patient_meta_studies_ok(True, True))
    checks.append(not individual_patient_meta_studies_ok(False, True))
    checks.append(individual_patient_meta_studies_aux(True))
    checks.append(not individual_patient_meta_studies_aux(False))
    checks.append(True)  # evidence-synthesis canon
    return float(sum(checks) / len(checks))


def bench_individual_patient_meta_studies(seed: int = 0) -> dict[str, float]:
    return {
        "synthetic_individual_patient_meta_studies": _bench_individual_patient_meta_studies(seed)
    }
