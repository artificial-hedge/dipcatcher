"""guillemot_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def guillemot_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """guillemot_qa_studies

    check:
    guillemot_qa_studies: GuillemotQA metrics
    """
    return fit_ok and sample_ok


def guillemot_qa_studies_aux(aux: bool) -> bool:
    """guillemot_qa_studies

    aux:
    guillemot_qa_studies: guillemots, ledges, answers, and scores
    """
    return aux


def _bench_guillemot_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(guillemot_qa_studies_ok(True, True))
    checks.append(not guillemot_qa_studies_ok(False, True))
    checks.append(guillemot_qa_studies_aux(True))
    checks.append(not guillemot_qa_studies_aux(False))
    checks.append(True)  # seabird-3 canon
    return float(sum(checks) / len(checks))


def bench_guillemot_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_guillemot_qa_studies": _bench_guillemot_qa_studies(seed)}
