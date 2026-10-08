"""Wave-1261 bench adapters: trial-statistics/HEOR canon (SYNTHETIC only)."""

from quant_fund.models import (
    biostatistics_methods_studies,
    epidemiology_methods_studies,
    heor_studies,
    regulatory_science_studies,
    survival_trial_studies,
    translational_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12610


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


def bench_biostatistics_methods_studies_family(seed: int = _SEED + 0):
    """biostatistics_methods_studies: synthetic correctness bench."""
    return _finite_blob(biostatistics_methods_studies.bench_biostatistics_methods_studies(seed))


def bench_epidemiology_methods_studies_family(seed: int = _SEED + 1):
    """epidemiology_methods_studies: synthetic correctness bench."""
    return _finite_blob(epidemiology_methods_studies.bench_epidemiology_methods_studies(seed))


def bench_heor_studies_family(seed: int = _SEED + 2):
    """heor_studies: synthetic correctness bench."""
    return _finite_blob(heor_studies.bench_heor_studies(seed))


def bench_regulatory_science_studies_family(seed: int = _SEED + 3):
    """regulatory_science_studies: synthetic correctness bench."""
    return _finite_blob(regulatory_science_studies.bench_regulatory_science_studies(seed))


def bench_survival_trial_studies_family(seed: int = _SEED + 4):
    """survival_trial_studies: synthetic correctness bench."""
    return _finite_blob(survival_trial_studies.bench_survival_trial_studies(seed))


def bench_translational_studies_family(seed: int = _SEED + 5):
    """translational_studies: synthetic correctness bench."""
    return _finite_blob(translational_studies.bench_translational_studies(seed))
