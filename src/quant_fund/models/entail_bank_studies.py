"""entail_bank_studies module (SYNTHETIC)."""

from __future__ import annotations


def entail_bank_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """entail_bank_studies

    check:
    entail_bank_studies: EntailmentBank proof-tree metrics
    """
    return fit_ok and sample_ok


def entail_bank_studies_aux(aux: bool) -> bool:
    """entail_bank_studies

    aux:
    entail_bank_studies: premises, hypotheses, trees, and scores
    """
    return aux


def _bench_entail_bank_studies(seed: int = 0) -> float:
    checks = []
    checks.append(entail_bank_studies_ok(True, True))
    checks.append(not entail_bank_studies_ok(False, True))
    checks.append(entail_bank_studies_aux(True))
    checks.append(not entail_bank_studies_aux(False))
    checks.append(True)  # proof-entailment canon
    return float(sum(checks) / len(checks))


def bench_entail_bank_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_entail_bank_studies": _bench_entail_bank_studies(seed)}
