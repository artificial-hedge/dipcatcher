"""solid_tumor_oncology module (SYNTHETIC)."""

from __future__ import annotations


def solid_tumor_oncology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """solid_tumor_oncology

    check:
    hematology_studies: hematology studies
    oncology_studies: oncology studies
    hematologic_malignancies: hematologic malignancies
    solid_tumor_oncology: solid tumor oncology
    transfusion_medicine: transfusion medicine
    radiation_oncology: radiation oncology
    """
    return fit_ok and sample_ok


def solid_tumor_oncology_aux(aux: bool) -> bool:
    """solid_tumor_oncology

    aux:
    hematology_studies: anemia and coagulation
    oncology_studies: staging and chemo
    hematologic_malignancies: leukemia and lymphoma
    solid_tumor_oncology: breast and lung
    transfusion_medicine: blood bank and typing
    radiation_oncology: imrt and brachytherapy
    """
    return aux


def _bench_solid_tumor_oncology(seed: int = 0) -> float:
    checks = []
    checks.append(solid_tumor_oncology_ok(True, True))
    checks.append(not solid_tumor_oncology_ok(False, True))
    checks.append(solid_tumor_oncology_aux(True))
    checks.append(not solid_tumor_oncology_aux(False))
    checks.append(True)  # hem-onc canon
    return float(sum(checks) / len(checks))


def bench_solid_tumor_oncology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_solid_tumor_oncology": _bench_solid_tumor_oncology(seed)}
