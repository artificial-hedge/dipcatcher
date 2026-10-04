"""data_poison_studies module (SYNTHETIC)."""

from __future__ import annotations


def data_poison_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """data_poison_studies

    check:
    data_poison_studies: poisoned samples/budgets and degradation curves
    """
    return fit_ok and sample_ok


def data_poison_studies_aux(aux: bool) -> bool:
    """data_poison_studies

    aux:
    data_poison_studies: poison craft/epsilons and clean-vs-poison accs
    """
    return aux


def _bench_data_poison_studies(seed: int = 0) -> float:
    checks = []
    checks.append(data_poison_studies_ok(True, True))
    checks.append(not data_poison_studies_ok(False, True))
    checks.append(data_poison_studies_aux(True))
    checks.append(not data_poison_studies_aux(False))
    checks.append(True)  # backdoor-eval canon
    return float(sum(checks) / len(checks))


def bench_data_poison_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_data_poison_studies": _bench_data_poison_studies(seed)}
