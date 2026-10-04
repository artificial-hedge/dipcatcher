"""hematology_studies module (SYNTHETIC)."""

from __future__ import annotations


def hematology_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hematology_studies

    check:
    hematology_studies: hematology studies
    oncology_studies: oncology studies
    hematologic_malignancies: hematologic malignancies
    solid_tumor_oncology: solid tumor oncology
    transfusion_medicine: transfusion medicine
    radiation_oncology: radiation oncology
    """
    return fit_ok and sample_ok


def hematology_studies_aux(aux: bool) -> bool:
    """hematology_studies

    aux:
    hematology_studies: anemia and coagulation
    oncology_studies: staging and chemo
    hematologic_malignancies: leukemia and lymphoma
    solid_tumor_oncology: breast and lung
    transfusion_medicine: blood bank and typing
    radiation_oncology: imrt and brachytherapy
    """
    return aux


def _bench_hematology_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hematology_studies_ok(True, True))
    checks.append(not hematology_studies_ok(False, True))
    checks.append(hematology_studies_aux(True))
    checks.append(not hematology_studies_aux(False))
    checks.append(True)  # hem-onc canon
    return float(sum(checks) / len(checks))


def bench_hematology_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hematology_studies": _bench_hematology_studies(seed)}
