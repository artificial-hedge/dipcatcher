"""Wave-1276 bench adapters: interpretability-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    attribution_patching_studies,
    causal_scrubbing_studies,
    function_vector_studies,
    induction_head_studies,
    monosemantic_studies,
    superposition_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12760


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


def bench_attribution_patching_studies_family(seed: int = _SEED + 0):
    """attribution_patching_studies: synthetic correctness bench."""
    return _finite_blob(attribution_patching_studies.bench_attribution_patching_studies(seed))


def bench_causal_scrubbing_studies_family(seed: int = _SEED + 1):
    """causal_scrubbing_studies: synthetic correctness bench."""
    return _finite_blob(causal_scrubbing_studies.bench_causal_scrubbing_studies(seed))


def bench_function_vector_studies_family(seed: int = _SEED + 2):
    """function_vector_studies: synthetic correctness bench."""
    return _finite_blob(function_vector_studies.bench_function_vector_studies(seed))


def bench_induction_head_studies_family(seed: int = _SEED + 3):
    """induction_head_studies: synthetic correctness bench."""
    return _finite_blob(induction_head_studies.bench_induction_head_studies(seed))


def bench_monosemantic_studies_family(seed: int = _SEED + 4):
    """monosemantic_studies: synthetic correctness bench."""
    return _finite_blob(monosemantic_studies.bench_monosemantic_studies(seed))


def bench_superposition_studies_family(seed: int = _SEED + 5):
    """superposition_studies: synthetic correctness bench."""
    return _finite_blob(superposition_studies.bench_superposition_studies(seed))
