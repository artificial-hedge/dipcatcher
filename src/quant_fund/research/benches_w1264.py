"""Wave-1264 bench adapters: causal-RWE-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    external_control_studies,
    negative_control_studies,
    probabilistic_bias_studies,
    self_controlled_studies,
    structural_nested_studies,
    transportability_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12640


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


def bench_external_control_studies_family(seed: int = _SEED + 0):
    """external_control_studies: synthetic correctness bench."""
    return _finite_blob(external_control_studies.bench_external_control_studies(seed))


def bench_negative_control_studies_family(seed: int = _SEED + 1):
    """negative_control_studies: synthetic correctness bench."""
    return _finite_blob(negative_control_studies.bench_negative_control_studies(seed))


def bench_probabilistic_bias_studies_family(seed: int = _SEED + 2):
    """probabilistic_bias_studies: synthetic correctness bench."""
    return _finite_blob(probabilistic_bias_studies.bench_probabilistic_bias_studies(seed))


def bench_self_controlled_studies_family(seed: int = _SEED + 3):
    """self_controlled_studies: synthetic correctness bench."""
    return _finite_blob(self_controlled_studies.bench_self_controlled_studies(seed))


def bench_structural_nested_studies_family(seed: int = _SEED + 4):
    """structural_nested_studies: synthetic correctness bench."""
    return _finite_blob(structural_nested_studies.bench_structural_nested_studies(seed))


def bench_transportability_studies_family(seed: int = _SEED + 5):
    """transportability_studies: synthetic correctness bench."""
    return _finite_blob(transportability_studies.bench_transportability_studies(seed))
