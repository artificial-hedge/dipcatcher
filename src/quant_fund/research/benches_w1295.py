"""Wave-1295 bench adapters: mech-anomaly/jailbreak canon (SYNTHETIC only)."""

from quant_fund.models import (
    activation_patch_studies,
    circuit_tracer_studies,
    feature_dashboard_studies,
    jailbreak_detect_studies,
    mech_anomaly_studies,
    sae_linter_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12950


def _finite_blob(blob):
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_activation_patch_studies_family(seed: int = _SEED + 0):
    """activation_patch_studies: synthetic correctness bench."""
    return _finite_blob(activation_patch_studies.bench_activation_patch_studies(seed))


def bench_circuit_tracer_studies_family(seed: int = _SEED + 1):
    """circuit_tracer_studies: synthetic correctness bench."""
    return _finite_blob(circuit_tracer_studies.bench_circuit_tracer_studies(seed))


def bench_feature_dashboard_studies_family(seed: int = _SEED + 2):
    """feature_dashboard_studies: synthetic correctness bench."""
    return _finite_blob(feature_dashboard_studies.bench_feature_dashboard_studies(seed))


def bench_jailbreak_detect_studies_family(seed: int = _SEED + 3):
    """jailbreak_detect_studies: synthetic correctness bench."""
    return _finite_blob(jailbreak_detect_studies.bench_jailbreak_detect_studies(seed))


def bench_mech_anomaly_studies_family(seed: int = _SEED + 4):
    """mech_anomaly_studies: synthetic correctness bench."""
    return _finite_blob(mech_anomaly_studies.bench_mech_anomaly_studies(seed))


def bench_sae_linter_studies_family(seed: int = _SEED + 5):
    """sae_linter_studies: synthetic correctness bench."""
    return _finite_blob(sae_linter_studies.bench_sae_linter_studies(seed))
