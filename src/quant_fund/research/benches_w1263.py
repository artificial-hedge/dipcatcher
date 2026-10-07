"""Wave-1263 bench adapters: evidence-synthesis canon (SYNTHETIC only)."""

from quant_fund.models import (
    diagnostic_meta_studies,
    fragility_index_studies,
    individual_patient_meta_studies,
    network_meta_studies,
    trial_sequential_studies,
    umbrella_review_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12630


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


def bench_diagnostic_meta_studies_family(seed: int = _SEED + 0):
    """diagnostic_meta_studies: synthetic correctness bench."""
    return _finite_blob(diagnostic_meta_studies.bench_diagnostic_meta_studies(seed))


def bench_fragility_index_studies_family(seed: int = _SEED + 1):
    """fragility_index_studies: synthetic correctness bench."""
    return _finite_blob(fragility_index_studies.bench_fragility_index_studies(seed))


def bench_individual_patient_meta_studies_family(seed: int = _SEED + 2):
    """individual_patient_meta_studies: synthetic correctness bench."""
    return _finite_blob(individual_patient_meta_studies.bench_individual_patient_meta_studies(seed))


def bench_network_meta_studies_family(seed: int = _SEED + 3):
    """network_meta_studies: synthetic correctness bench."""
    return _finite_blob(network_meta_studies.bench_network_meta_studies(seed))


def bench_trial_sequential_studies_family(seed: int = _SEED + 4):
    """trial_sequential_studies: synthetic correctness bench."""
    return _finite_blob(trial_sequential_studies.bench_trial_sequential_studies(seed))


def bench_umbrella_review_studies_family(seed: int = _SEED + 5):
    """umbrella_review_studies: synthetic correctness bench."""
    return _finite_blob(umbrella_review_studies.bench_umbrella_review_studies(seed))
