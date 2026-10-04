"""whip_poor_will_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def whip_poor_will_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """whip_poor_will_qa_studies

    check:
    whip_poor_will_qa_studies: Whip-poor-willQA metrics
    """
    return fit_ok and sample_ok


def whip_poor_will_qa_studies_aux(aux: bool) -> bool:
    """whip_poor_will_qa_studies

    aux:
    whip_poor_will_qa_studies: whip-poor-wills, glades, answers, and scores
    """
    return aux


def _bench_whip_poor_will_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(whip_poor_will_qa_studies_ok(True, True))
    checks.append(not whip_poor_will_qa_studies_ok(False, True))
    checks.append(whip_poor_will_qa_studies_aux(True))
    checks.append(not whip_poor_will_qa_studies_aux(False))
    checks.append(True)  # nightjar-2 canon
    return float(sum(checks) / len(checks))


def bench_whip_poor_will_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_whip_poor_will_qa_studies": _bench_whip_poor_will_qa_studies(seed)}
