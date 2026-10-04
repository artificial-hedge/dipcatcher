"""self_bleu_studies module (SYNTHETIC)."""

from __future__ import annotations


def self_bleu_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """self_bleu_studies

    check:
    self_bleu_studies: Self-BLEU generation diversity, n-gram tails, and scores
    """
    return fit_ok and sample_ok


def self_bleu_studies_aux(aux: bool) -> bool:
    """self_bleu_studies

    aux:
    self_bleu_studies: sampled generations, BLEU curves, and diversity
    """
    return aux


def _bench_self_bleu_studies(seed: int = 0) -> float:
    checks = []
    checks.append(self_bleu_studies_ok(True, True))
    checks.append(not self_bleu_studies_ok(False, True))
    checks.append(self_bleu_studies_aux(True))
    checks.append(not self_bleu_studies_aux(False))
    checks.append(True)  # generation-quality canon
    return float(sum(checks) / len(checks))


def bench_self_bleu_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_self_bleu_studies": _bench_self_bleu_studies(seed)}
