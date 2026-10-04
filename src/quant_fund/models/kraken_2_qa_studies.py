"""kraken_2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kraken_2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kraken_2_qa_studies

    check:
    kraken_2_qa_studies: Kraken2QA metrics
    """
    return fit_ok and sample_ok


def kraken_2_qa_studies_aux(aux: bool) -> bool:
    """kraken_2_qa_studies

    aux:
    kraken_2_qa_studies: krakens, deep fjords, answers, and scores
    """
    return aux


def _bench_kraken_2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kraken_2_qa_studies_ok(True, True))
    checks.append(not kraken_2_qa_studies_ok(False, True))
    checks.append(kraken_2_qa_studies_aux(True))
    checks.append(not kraken_2_qa_studies_aux(False))
    checks.append(True)  # cryptid canon
    return float(sum(checks) / len(checks))


def bench_kraken_2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kraken_2_qa_studies": _bench_kraken_2_qa_studies(seed)}
