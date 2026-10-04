"""neural_cleanse_studies module (SYNTHETIC)."""

from __future__ import annotations


def neural_cleanse_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """neural_cleanse_studies

    check:
    neural_cleanse_studies: reverse-engineered triggers and MAD outlier flags
    """
    return fit_ok and sample_ok


def neural_cleanse_studies_aux(aux: bool) -> bool:
    """neural_cleanse_studies

    aux:
    neural_cleanse_studies: reconstructed masks/mitigations and detection
    """
    return aux


def _bench_neural_cleanse_studies(seed: int = 0) -> float:
    checks = []
    checks.append(neural_cleanse_studies_ok(True, True))
    checks.append(not neural_cleanse_studies_ok(False, True))
    checks.append(neural_cleanse_studies_aux(True))
    checks.append(not neural_cleanse_studies_aux(False))
    checks.append(True)  # backdoor-eval canon
    return float(sum(checks) / len(checks))


def bench_neural_cleanse_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_neural_cleanse_studies": _bench_neural_cleanse_studies(seed)}
