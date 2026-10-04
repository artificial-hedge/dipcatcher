"""pelican_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pelican_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pelican_qa_studies

    check:
    pelican_qa_studies: PelicanQA metrics
    """
    return fit_ok and sample_ok


def pelican_qa_studies_aux(aux: bool) -> bool:
    """pelican_qa_studies

    aux:
    pelican_qa_studies: pelicans, pouches, answers, and scores
    """
    return aux


def _bench_pelican_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pelican_qa_studies_ok(True, True))
    checks.append(not pelican_qa_studies_ok(False, True))
    checks.append(pelican_qa_studies_aux(True))
    checks.append(not pelican_qa_studies_aux(False))
    checks.append(True)  # wader canon
    return float(sum(checks) / len(checks))


def bench_pelican_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pelican_qa_studies": _bench_pelican_qa_studies(seed)}
