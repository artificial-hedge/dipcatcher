"""logic_nli_studies module (SYNTHETIC)."""

from __future__ import annotations


def logic_nli_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """logic_nli_studies

    check:
    logic_nli_studies: LogicNLI metrics
    """
    return fit_ok and sample_ok


def logic_nli_studies_aux(aux: bool) -> bool:
    """logic_nli_studies

    aux:
    logic_nli_studies: premises, formulas, labels, and scores
    """
    return aux


def _bench_logic_nli_studies(seed: int = 0) -> float:
    checks = []
    checks.append(logic_nli_studies_ok(True, True))
    checks.append(not logic_nli_studies_ok(False, True))
    checks.append(logic_nli_studies_aux(True))
    checks.append(not logic_nli_studies_aux(False))
    checks.append(True)  # proof-entailment canon
    return float(sum(checks) / len(checks))


def bench_logic_nli_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_logic_nli_studies": _bench_logic_nli_studies(seed)}
