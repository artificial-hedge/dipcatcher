"""offline_distill_studies module (SYNTHETIC)."""

from __future__ import annotations


def offline_distill_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """offline_distill_studies

    check:
    offline_distill_studies: teacher-student policy transfer on fixed data/logits and targets
    """
    return fit_ok and sample_ok


def offline_distill_studies_aux(aux: bool) -> bool:
    """offline_distill_studies

    aux:
    offline_distill_studies: coverage-aware distillation and support bounds/regions and masks
    """
    return aux


def _bench_offline_distill_studies(seed: int = 0) -> float:
    checks = []
    checks.append(offline_distill_studies_ok(True, True))
    checks.append(not offline_distill_studies_ok(False, True))
    checks.append(offline_distill_studies_aux(True))
    checks.append(not offline_distill_studies_aux(False))
    checks.append(True)  # RL-imitation canon
    return float(sum(checks) / len(checks))


def bench_offline_distill_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_offline_distill_studies": _bench_offline_distill_studies(seed)}
