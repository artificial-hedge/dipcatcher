"""Wave-1269 bench adapters: post-training-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    best_of_n_studies,
    cdpo_studies,
    constitutional_ai_studies,
    orpo_studies,
    simpo_studies,
    sppo_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12690


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


def bench_best_of_n_studies_family(seed: int = _SEED + 0):
    """best_of_n_studies: synthetic correctness bench."""
    return _finite_blob(best_of_n_studies.bench_best_of_n_studies(seed))


def bench_cdpo_studies_family(seed: int = _SEED + 1):
    """cdpo_studies: synthetic correctness bench."""
    return _finite_blob(cdpo_studies.bench_cdpo_studies(seed))


def bench_constitutional_ai_studies_family(seed: int = _SEED + 2):
    """constitutional_ai_studies: synthetic correctness bench."""
    return _finite_blob(constitutional_ai_studies.bench_constitutional_ai_studies(seed))


def bench_orpo_studies_family(seed: int = _SEED + 3):
    """orpo_studies: synthetic correctness bench."""
    return _finite_blob(orpo_studies.bench_orpo_studies(seed))


def bench_simpo_studies_family(seed: int = _SEED + 4):
    """simpo_studies: synthetic correctness bench."""
    return _finite_blob(simpo_studies.bench_simpo_studies(seed))


def bench_sppo_studies_family(seed: int = _SEED + 5):
    """sppo_studies: synthetic correctness bench."""
    return _finite_blob(sppo_studies.bench_sppo_studies(seed))
