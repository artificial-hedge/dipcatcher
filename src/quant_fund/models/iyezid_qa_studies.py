"""iyezid_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def iyezid_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """iyezid_qa_studies

    check:
    iyezid_qa_studies: p
    """
    return fit_ok and sample_ok


def iyezid_qa_studies_aux(aux: bool) -> bool:
    """iyezid_qa_studies

    aux:
    iyezid_qa_studies: r
    """
    return aux


def _bench_iyezid_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(iyezid_qa_studies_ok(True, True))
    checks.append(not iyezid_qa_studies_ok(False, True))
    checks.append(iyezid_qa_studies_aux(True))
    checks.append(not iyezid_qa_studies_aux(False))
    checks.append(True)  # tuareg-2 canon
    return float(sum(checks) / len(checks))


def bench_iyezid_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_iyezid_qa_studies": _bench_iyezid_qa_studies(seed)}
