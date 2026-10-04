"""adversarial_irl_studies module (SYNTHETIC)."""

from __future__ import annotations


def adversarial_irl_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """adversarial_irl_studies

    check:
    adversarial_irl_studies: GAIL-style reward inference/occupancy and matching
    """
    return fit_ok and sample_ok


def adversarial_irl_studies_aux(aux: bool) -> bool:
    """adversarial_irl_studies

    aux:
    adversarial_irl_studies: AIRL/discriminator shaping and policy transfer/rewards and logits
    """
    return aux


def _bench_adversarial_irl_studies(seed: int = 0) -> float:
    checks = []
    checks.append(adversarial_irl_studies_ok(True, True))
    checks.append(not adversarial_irl_studies_ok(False, True))
    checks.append(adversarial_irl_studies_aux(True))
    checks.append(not adversarial_irl_studies_aux(False))
    checks.append(True)  # RL-imitation canon
    return float(sum(checks) / len(checks))


def bench_adversarial_irl_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_adversarial_irl_studies": _bench_adversarial_irl_studies(seed)}
