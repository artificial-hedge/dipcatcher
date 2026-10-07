"""Wave-1302 bench adapters: privacy-inference canon (SYNTHETIC only)."""

from quant_fund.models import (
    canary_infer_studies,
    deep_leak_studies,
    gradient_leak_studies,
    lira_studies,
    membership_infer_studies,
    shadow_model_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13020


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


def bench_canary_infer_studies_family(seed: int = _SEED + 0):
    """canary_infer_studies: synthetic correctness bench."""
    return _finite_blob(canary_infer_studies.bench_canary_infer_studies(seed))


def bench_deep_leak_studies_family(seed: int = _SEED + 1):
    """deep_leak_studies: synthetic correctness bench."""
    return _finite_blob(deep_leak_studies.bench_deep_leak_studies(seed))


def bench_gradient_leak_studies_family(seed: int = _SEED + 2):
    """gradient_leak_studies: synthetic correctness bench."""
    return _finite_blob(gradient_leak_studies.bench_gradient_leak_studies(seed))


def bench_lira_studies_family(seed: int = _SEED + 3):
    """lira_studies: synthetic correctness bench."""
    return _finite_blob(lira_studies.bench_lira_studies(seed))


def bench_membership_infer_studies_family(seed: int = _SEED + 4):
    """membership_infer_studies: synthetic correctness bench."""
    return _finite_blob(membership_infer_studies.bench_membership_infer_studies(seed))


def bench_shadow_model_studies_family(seed: int = _SEED + 5):
    """shadow_model_studies: synthetic correctness bench."""
    return _finite_blob(shadow_model_studies.bench_shadow_model_studies(seed))
