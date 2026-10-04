"""ruby_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ruby_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ruby_qa_studies

    check:
    ruby_qa_studies: RubyQA metrics
    """
    return fit_ok and sample_ok


def ruby_qa_studies_aux(aux: bool) -> bool:
    """ruby_qa_studies

    aux:
    ruby_qa_studies: rubies, pigeons, answers, and scores
    """
    return aux


def _bench_ruby_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ruby_qa_studies_ok(True, True))
    checks.append(not ruby_qa_studies_ok(False, True))
    checks.append(ruby_qa_studies_aux(True))
    checks.append(not ruby_qa_studies_aux(False))
    checks.append(True)  # gemstone canon
    return float(sum(checks) / len(checks))


def bench_ruby_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ruby_qa_studies": _bench_ruby_qa_studies(seed)}
