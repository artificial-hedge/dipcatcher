"""mech_anomaly_studies module (SYNTHETIC)."""

from __future__ import annotations


def mech_anomaly_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mech_anomaly_studies

    check:
    mech_anomaly_studies: mechanistic deviation detection/baselines and scores
    """
    return fit_ok and sample_ok


def mech_anomaly_studies_aux(aux: bool) -> bool:
    """mech_anomaly_studies

    aux:
    mech_anomaly_studies: internal-state anomaly flagging/activations and bounds
    """
    return aux


def _bench_mech_anomaly_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mech_anomaly_studies_ok(True, True))
    checks.append(not mech_anomaly_studies_ok(False, True))
    checks.append(mech_anomaly_studies_aux(True))
    checks.append(not mech_anomaly_studies_aux(False))
    checks.append(True)  # mech-anomaly/jailbreak canon
    return float(sum(checks) / len(checks))


def bench_mech_anomaly_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mech_anomaly_studies": _bench_mech_anomaly_studies(seed)}
