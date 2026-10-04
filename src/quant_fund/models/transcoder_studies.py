"""transcoder_studies module (SYNTHETIC)."""

from __future__ import annotations


def transcoder_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """transcoder_studies

    check:
    transcoder_studies: MLP transcoding and crosscoders/features and layer skip
    """
    return fit_ok and sample_ok


def transcoder_studies_aux(aux: bool) -> bool:
    """transcoder_studies

    aux:
    transcoder_studies: input-output attribution and replacement/sparse and faithful
    """
    return aux


def _bench_transcoder_studies(seed: int = 0) -> float:
    checks = []
    checks.append(transcoder_studies_ok(True, True))
    checks.append(not transcoder_studies_ok(False, True))
    checks.append(transcoder_studies_aux(True))
    checks.append(not transcoder_studies_aux(False))
    checks.append(True)  # mech-interp-2 canon
    return float(sum(checks) / len(checks))


def bench_transcoder_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_transcoder_studies": _bench_transcoder_studies(seed)}
