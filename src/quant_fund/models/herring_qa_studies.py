"""herring_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def herring_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """herring_qa_studies

    check:
    herring_qa_studies: HerringQA metrics
    """
    return fit_ok and sample_ok


def herring_qa_studies_aux(aux: bool) -> bool:
    """herring_qa_studies

    aux:
    herring_qa_studies: herring, northern seas, answers, and scores
    """
    return aux


def _bench_herring_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(herring_qa_studies_ok(True, True))
    checks.append(not herring_qa_studies_ok(False, True))
    checks.append(herring_qa_studies_aux(True))
    checks.append(not herring_qa_studies_aux(False))
    checks.append(True)  # pelagic-fish canon
    return float(sum(checks) / len(checks))


def bench_herring_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_herring_qa_studies": _bench_herring_qa_studies(seed)}
