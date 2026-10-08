"""Wave-1314 bench adapters: privacy-inference canon (SYNTHETIC only)."""

from quant_fund.models import (
    attribute_infer_studies,
    extraction_studies,
    inversion_studies,
    model_stealing_studies,
    property_infer_studies,
    reconstruction_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13140


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


def bench_attribute_infer_studies_family(seed: int = _SEED + 0):
    """attribute_infer_studies: synthetic correctness bench."""
    return _finite_blob(attribute_infer_studies.bench_attribute_infer_studies(seed))


def bench_extraction_studies_family(seed: int = _SEED + 1):
    """extraction_studies: synthetic correctness bench."""
    return _finite_blob(extraction_studies.bench_extraction_studies(seed))


def bench_inversion_studies_family(seed: int = _SEED + 2):
    """inversion_studies: synthetic correctness bench."""
    return _finite_blob(inversion_studies.bench_inversion_studies(seed))


def bench_model_stealing_studies_family(seed: int = _SEED + 3):
    """model_stealing_studies: synthetic correctness bench."""
    return _finite_blob(model_stealing_studies.bench_model_stealing_studies(seed))


def bench_property_infer_studies_family(seed: int = _SEED + 4):
    """property_infer_studies: synthetic correctness bench."""
    return _finite_blob(property_infer_studies.bench_property_infer_studies(seed))


def bench_reconstruction_studies_family(seed: int = _SEED + 5):
    """reconstruction_studies: synthetic correctness bench."""
    return _finite_blob(reconstruction_studies.bench_reconstruction_studies(seed))
