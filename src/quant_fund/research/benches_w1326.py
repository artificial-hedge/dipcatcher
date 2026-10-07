"""Wave-1326 bench adapters: privacy-inference-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    attribute_inference_studies,
    canary_memorization_studies,
    extraction_attack_studies,
    membership_inference_studies,
    model_inversion_studies,
    privacy_meter_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13260


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


def bench_attribute_inference_studies_family(seed: int = _SEED + 0):
    """attribute_inference_studies: synthetic correctness bench."""
    return _finite_blob(attribute_inference_studies.bench_attribute_inference_studies(seed))


def bench_canary_memorization_studies_family(seed: int = _SEED + 1):
    """canary_memorization_studies: synthetic correctness bench."""
    return _finite_blob(canary_memorization_studies.bench_canary_memorization_studies(seed))


def bench_extraction_attack_studies_family(seed: int = _SEED + 2):
    """extraction_attack_studies: synthetic correctness bench."""
    return _finite_blob(extraction_attack_studies.bench_extraction_attack_studies(seed))


def bench_membership_inference_studies_family(seed: int = _SEED + 3):
    """membership_inference_studies: synthetic correctness bench."""
    return _finite_blob(membership_inference_studies.bench_membership_inference_studies(seed))


def bench_model_inversion_studies_family(seed: int = _SEED + 4):
    """model_inversion_studies: synthetic correctness bench."""
    return _finite_blob(model_inversion_studies.bench_model_inversion_studies(seed))


def bench_privacy_meter_studies_family(seed: int = _SEED + 5):
    """privacy_meter_studies: synthetic correctness bench."""
    return _finite_blob(privacy_meter_studies.bench_privacy_meter_studies(seed))
