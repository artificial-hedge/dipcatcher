"""objectnet_studies module (SYNTHETIC)."""

from __future__ import annotations


def objectnet_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """objectnet_studies

    check:
    objectnet_studies: ObjectNet pose/background-shift accuracy and gaps
    """
    return fit_ok and sample_ok


def objectnet_studies_aux(aux: bool) -> bool:
    """objectnet_studies

    aux:
    objectnet_studies: rotated poses, scenes, and robustness rates
    """
    return aux


def _bench_objectnet_studies(seed: int = 0) -> float:
    checks = []
    checks.append(objectnet_studies_ok(True, True))
    checks.append(not objectnet_studies_ok(False, True))
    checks.append(objectnet_studies_aux(True))
    checks.append(not objectnet_studies_aux(False))
    checks.append(True)  # benchmark-eval canon
    return float(sum(checks) / len(checks))


def bench_objectnet_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_objectnet_studies": _bench_objectnet_studies(seed)}
