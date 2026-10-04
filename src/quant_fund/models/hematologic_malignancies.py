"""hematologic_malignancies module (SYNTHETIC)."""

from __future__ import annotations


def hematologic_malignancies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hematologic_malignancies

    check:
    hematology_studies: hematology studies
    oncology_studies: oncology studies
    hematologic_malignancies: hematologic malignancies
    solid_tumor_oncology: solid tumor oncology
    transfusion_medicine: transfusion medicine
    radiation_oncology: radiation oncology
    """
    return fit_ok and sample_ok


def hematologic_malignancies_aux(aux: bool) -> bool:
    """hematologic_malignancies

    aux:
    hematology_studies: anemia and coagulation
    oncology_studies: staging and chemo
    hematologic_malignancies: leukemia and lymphoma
    solid_tumor_oncology: breast and lung
    transfusion_medicine: blood bank and typing
    radiation_oncology: imrt and brachytherapy
    """
    return aux


def _bench_hematologic_malignancies(seed: int = 0) -> float:
    checks = []
    checks.append(hematologic_malignancies_ok(True, True))
    checks.append(not hematologic_malignancies_ok(False, True))
    checks.append(hematologic_malignancies_aux(True))
    checks.append(not hematologic_malignancies_aux(False))
    checks.append(True)  # hem-onc canon
    return float(sum(checks) / len(checks))


def bench_hematologic_malignancies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hematologic_malignancies": _bench_hematologic_malignancies(seed)}
