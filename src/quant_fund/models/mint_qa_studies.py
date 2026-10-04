"""mint_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mint_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mint_qa_studies

    check:
    mint_qa_studies: MintQA metrics
    """
    return fit_ok and sample_ok


def mint_qa_studies_aux(aux: bool) -> bool:
    """mint_qa_studies

    aux:
    mint_qa_studies: mints, sprigs, answers, and scores
    """
    return aux


def _bench_mint_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mint_qa_studies_ok(True, True))
    checks.append(not mint_qa_studies_ok(False, True))
    checks.append(mint_qa_studies_aux(True))
    checks.append(not mint_qa_studies_aux(False))
    checks.append(True)  # spice canon
    return float(sum(checks) / len(checks))


def bench_mint_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mint_qa_studies": _bench_mint_qa_studies(seed)}
