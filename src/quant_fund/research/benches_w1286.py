"""Wave-1286 bench adapters: long-context canon (SYNTHETIC only)."""

from quant_fund.models import (
    beacon_context_studies,
    hierarchical_context_studies,
    infini_attention_studies,
    landmark_attention_studies,
    ntk_scaling_studies,
    yarn_scaling_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12860


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


def bench_beacon_context_studies_family(seed: int = _SEED + 0):
    """beacon_context_studies: synthetic correctness bench."""
    return _finite_blob(beacon_context_studies.bench_beacon_context_studies(seed))


def bench_hierarchical_context_studies_family(seed: int = _SEED + 1):
    """hierarchical_context_studies: synthetic correctness bench."""
    return _finite_blob(hierarchical_context_studies.bench_hierarchical_context_studies(seed))


def bench_infini_attention_studies_family(seed: int = _SEED + 2):
    """infini_attention_studies: synthetic correctness bench."""
    return _finite_blob(infini_attention_studies.bench_infini_attention_studies(seed))


def bench_landmark_attention_studies_family(seed: int = _SEED + 3):
    """landmark_attention_studies: synthetic correctness bench."""
    return _finite_blob(landmark_attention_studies.bench_landmark_attention_studies(seed))


def bench_ntk_scaling_studies_family(seed: int = _SEED + 4):
    """ntk_scaling_studies: synthetic correctness bench."""
    return _finite_blob(ntk_scaling_studies.bench_ntk_scaling_studies(seed))


def bench_yarn_scaling_studies_family(seed: int = _SEED + 5):
    """yarn_scaling_studies: synthetic correctness bench."""
    return _finite_blob(yarn_scaling_studies.bench_yarn_scaling_studies(seed))
