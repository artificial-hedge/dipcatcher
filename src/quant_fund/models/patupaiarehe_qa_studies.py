"""patupaiarehe_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def patupaiarehe_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """patupaiarehe_qa_studies

    check:
    patupaiarehe_qa_studies: P
    """
    return fit_ok and sample_ok


def patupaiarehe_qa_studies_aux(aux: bool) -> bool:
    """patupaiarehe_qa_studies

    aux:
    patupaiarehe_qa_studies: a
    """
    return aux


def _bench_patupaiarehe_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(patupaiarehe_qa_studies_ok(True, True))
    checks.append(not patupaiarehe_qa_studies_ok(False, True))
    checks.append(patupaiarehe_qa_studies_aux(True))
    checks.append(not patupaiarehe_qa_studies_aux(False))
    checks.append(True)  # polynesian-demon canon
    return float(sum(checks) / len(checks))


def bench_patupaiarehe_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_patupaiarehe_qa_studies": _bench_patupaiarehe_qa_studies(seed)}
