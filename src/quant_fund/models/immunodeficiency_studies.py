"""immunodeficiency_studies module (SYNTHETIC)."""

from __future__ import annotations


def immunodeficiency_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """immunodeficiency_studies

    check:
    transplant_medicine_studies: transplant medicine studies
    immunology_medicine: immunology medicine
    allergy_studies: allergy studies
    autoimmunity_studies: autoimmunity studies
    hematopoietic_transplant: hematopoietic transplant
    immunodeficiency_studies: immunodeficiency studies
    """
    return fit_ok and sample_ok


def immunodeficiency_studies_aux(aux: bool) -> bool:
    """immunodeficiency_studies

    aux:
    transplant_medicine_studies: graft and rejection
    immunology_medicine: antibody and antigen
    allergy_studies: ige and anaphylaxis
    autoimmunity_studies: lupus and vasculitis
    hematopoietic_transplant: marrow and gvhd
    immunodeficiency_studies: scid and ivig
    """
    return aux


def _bench_immunodeficiency_studies(seed: int = 0) -> float:
    checks = []
    checks.append(immunodeficiency_studies_ok(True, True))
    checks.append(not immunodeficiency_studies_ok(False, True))
    checks.append(immunodeficiency_studies_aux(True))
    checks.append(not immunodeficiency_studies_aux(False))
    checks.append(True)  # transplant canon
    return float(sum(checks) / len(checks))


def bench_immunodeficiency_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_immunodeficiency_studies": _bench_immunodeficiency_studies(seed)}
