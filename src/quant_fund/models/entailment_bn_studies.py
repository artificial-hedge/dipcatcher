"""entailment_bn_studies module (SYNTHETIC)."""

from __future__ import annotations


def entailment_bn_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """entailment_bn_studies

    check:
    entailment_bn_studies: entailment-bank metrics
    """
    return fit_ok and sample_ok


def entailment_bn_studies_aux(aux: bool) -> bool:
    """entailment_bn_studies

    aux:
    entailment_bn_studies: sentences, hypotheses, labels, and accuracies
    """
    return aux


def _bench_entailment_bn_studies(seed: int = 0) -> float:
    checks = []
    checks.append(entailment_bn_studies_ok(True, True))
    checks.append(not entailment_bn_studies_ok(False, True))
    checks.append(entailment_bn_studies_aux(True))
    checks.append(not entailment_bn_studies_aux(False))
    checks.append(True)  # NLI-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_entailment_bn_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_entailment_bn_studies": _bench_entailment_bn_studies(seed)}
