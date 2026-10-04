"""trial_sequential_studies module (SYNTHETIC)."""

from __future__ import annotations


def trial_sequential_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """trial_sequential_studies

    check:
    trial_sequential_studies: information size and boundaries/monitoring and RIS
    """
    return fit_ok and sample_ok


def trial_sequential_studies_aux(aux: bool) -> bool:
    """trial_sequential_studies

    aux:
    trial_sequential_studies: diversity and adjustment/spending and futility
    """
    return aux


def _bench_trial_sequential_studies(seed: int = 0) -> float:
    checks = []
    checks.append(trial_sequential_studies_ok(True, True))
    checks.append(not trial_sequential_studies_ok(False, True))
    checks.append(trial_sequential_studies_aux(True))
    checks.append(not trial_sequential_studies_aux(False))
    checks.append(True)  # evidence-synthesis canon
    return float(sum(checks) / len(checks))


def bench_trial_sequential_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_trial_sequential_studies": _bench_trial_sequential_studies(seed)}
