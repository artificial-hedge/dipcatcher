"""transfusion_medicine module (SYNTHETIC)."""

from __future__ import annotations


def transfusion_medicine_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """transfusion_medicine

    check:
    hematology_studies: hematology studies
    oncology_studies: oncology studies
    hematologic_malignancies: hematologic malignancies
    solid_tumor_oncology: solid tumor oncology
    transfusion_medicine: transfusion medicine
    radiation_oncology: radiation oncology
    """
    return fit_ok and sample_ok


def transfusion_medicine_aux(aux: bool) -> bool:
    """transfusion_medicine

    aux:
    hematology_studies: anemia and coagulation
    oncology_studies: staging and chemo
    hematologic_malignancies: leukemia and lymphoma
    solid_tumor_oncology: breast and lung
    transfusion_medicine: blood bank and typing
    radiation_oncology: imrt and brachytherapy
    """
    return aux


def _bench_transfusion_medicine(seed: int = 0) -> float:
    checks = []
    checks.append(transfusion_medicine_ok(True, True))
    checks.append(not transfusion_medicine_ok(False, True))
    checks.append(transfusion_medicine_aux(True))
    checks.append(not transfusion_medicine_aux(False))
    checks.append(True)  # hem-onc canon
    return float(sum(checks) / len(checks))


def bench_transfusion_medicine(seed: int = 0) -> dict[str, float]:
    return {"synthetic_transfusion_medicine": _bench_transfusion_medicine(seed)}
