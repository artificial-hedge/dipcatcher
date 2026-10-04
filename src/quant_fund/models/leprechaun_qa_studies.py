"""leprechaun_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def leprechaun_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """leprechaun_qa_studies

    check:
    leprechaun_qa_studies: LeprechaunQA metrics
    """
    return fit_ok and sample_ok


def leprechaun_qa_studies_aux(aux: bool) -> bool:
    """leprechaun_qa_studies

    aux:
    leprechaun_qa_studies: leprechauns, gold hoarders, answers, and scores
    """
    return aux


def _bench_leprechaun_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(leprechaun_qa_studies_ok(True, True))
    checks.append(not leprechaun_qa_studies_ok(False, True))
    checks.append(leprechaun_qa_studies_aux(True))
    checks.append(not leprechaun_qa_studies_aux(False))
    checks.append(True)  # celtic-beast canon
    return float(sum(checks) / len(checks))


def bench_leprechaun_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_leprechaun_qa_studies": _bench_leprechaun_qa_studies(seed)}
