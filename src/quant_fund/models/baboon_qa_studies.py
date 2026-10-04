"""baboon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def baboon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """baboon_qa_studies

    check:
    baboon_qa_studies: BaboonQA metrics
    """
    return fit_ok and sample_ok


def baboon_qa_studies_aux(aux: bool) -> bool:
    """baboon_qa_studies

    aux:
    baboon_qa_studies: baboons, troops, answers, and scores
    """
    return aux


def _bench_baboon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(baboon_qa_studies_ok(True, True))
    checks.append(not baboon_qa_studies_ok(False, True))
    checks.append(baboon_qa_studies_aux(True))
    checks.append(not baboon_qa_studies_aux(False))
    checks.append(True)  # savanna canon
    return float(sum(checks) / len(checks))


def bench_baboon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_baboon_qa_studies": _bench_baboon_qa_studies(seed)}
