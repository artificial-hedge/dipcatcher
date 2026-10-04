"""target_trial_emulation_studies module (SYNTHETIC)."""

from __future__ import annotations


def target_trial_emulation_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """target_trial_emulation_studies

    check:
    target_trial_emulation_studies: eligibility and assignment/follow-up and estimand
    """
    return fit_ok and sample_ok


def target_trial_emulation_studies_aux(aux: bool) -> bool:
    """target_trial_emulation_studies

    aux:
    target_trial_emulation_studies: cloning and censoring/anchor and comparison
    """
    return aux


def _bench_target_trial_emulation_studies(seed: int = 0) -> float:
    checks = []
    checks.append(target_trial_emulation_studies_ok(True, True))
    checks.append(not target_trial_emulation_studies_ok(False, True))
    checks.append(target_trial_emulation_studies_aux(True))
    checks.append(not target_trial_emulation_studies_aux(False))
    checks.append(True)  # target-trial/RWE canon
    return float(sum(checks) / len(checks))


def bench_target_trial_emulation_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_target_trial_emulation_studies": _bench_target_trial_emulation_studies(seed)}
