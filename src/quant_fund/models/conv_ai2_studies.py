"""conv_ai2_studies module (SYNTHETIC)."""

from __future__ import annotations


def conv_ai2_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """conv_ai2_studies

    check:
    conv_ai2_studies: ConvAI2 chit-chat metrics
    """
    return fit_ok and sample_ok


def conv_ai2_studies_aux(aux: bool) -> bool:
    """conv_ai2_studies

    aux:
    conv_ai2_studies: contexts, responses, references, and scores
    """
    return aux


def _bench_conv_ai2_studies(seed: int = 0) -> float:
    checks = []
    checks.append(conv_ai2_studies_ok(True, True))
    checks.append(not conv_ai2_studies_ok(False, True))
    checks.append(conv_ai2_studies_aux(True))
    checks.append(not conv_ai2_studies_aux(False))
    checks.append(True)  # dialogue-system canon
    return float(sum(checks) / len(checks))


def bench_conv_ai2_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_conv_ai2_studies": _bench_conv_ai2_studies(seed)}
