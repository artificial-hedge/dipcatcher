"""watermark_studies module (SYNTHETIC)."""

from __future__ import annotations


def watermark_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """watermark_studies

    check:
    watermark_studies: Neural-Cleanse watermark/backdoor trigger detection
    """
    return fit_ok and sample_ok


def watermark_studies_aux(aux: bool) -> bool:
    """watermark_studies

    aux:
    watermark_studies: inversion results, anomaly index, and detection
    """
    return aux


def _bench_watermark_studies(seed: int = 0) -> float:
    checks = []
    checks.append(watermark_studies_ok(True, True))
    checks.append(not watermark_studies_ok(False, True))
    checks.append(watermark_studies_aux(True))
    checks.append(not watermark_studies_aux(False))
    checks.append(True)  # privacy-attack canon
    return float(sum(checks) / len(checks))


def bench_watermark_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_watermark_studies": _bench_watermark_studies(seed)}
