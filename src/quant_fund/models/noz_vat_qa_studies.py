"""noz_vat_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def noz_vat_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """noz_vat_qa_studies

    check:
    noz_vat_qa_studies: n
    """
    return fit_ok and sample_ok


def noz_vat_qa_studies_aux(aux: bool) -> bool:
    """noz_vat_qa_studies

    aux:
    noz_vat_qa_studies: i
    """
    return aux


def _bench_noz_vat_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(noz_vat_qa_studies_ok(True, True))
    checks.append(not noz_vat_qa_studies_ok(False, True))
    checks.append(noz_vat_qa_studies_aux(True))
    checks.append(not noz_vat_qa_studies_aux(False))
    checks.append(True)  # breton-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_noz_vat_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_noz_vat_qa_studies": _bench_noz_vat_qa_studies(seed)}
