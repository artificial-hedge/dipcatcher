"""Wave-1319 bench adapters: benchmark-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    math_bench_studies,
    multirc_studies,
    ninco_studies,
    objectnet_studies,
    ood_bench_studies,
    wild_bench_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13190


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


def bench_math_bench_studies_family(seed: int = _SEED + 0):
    """math_bench_studies: synthetic correctness bench."""
    return _finite_blob(math_bench_studies.bench_math_bench_studies(seed))


def bench_multirc_studies_family(seed: int = _SEED + 1):
    """multirc_studies: synthetic correctness bench."""
    return _finite_blob(multirc_studies.bench_multirc_studies(seed))


def bench_ninco_studies_family(seed: int = _SEED + 2):
    """ninco_studies: synthetic correctness bench."""
    return _finite_blob(ninco_studies.bench_ninco_studies(seed))


def bench_objectnet_studies_family(seed: int = _SEED + 3):
    """objectnet_studies: synthetic correctness bench."""
    return _finite_blob(objectnet_studies.bench_objectnet_studies(seed))


def bench_ood_bench_studies_family(seed: int = _SEED + 4):
    """ood_bench_studies: synthetic correctness bench."""
    return _finite_blob(ood_bench_studies.bench_ood_bench_studies(seed))


def bench_wild_bench_studies_family(seed: int = _SEED + 5):
    """wild_bench_studies: synthetic correctness bench."""
    return _finite_blob(wild_bench_studies.bench_wild_bench_studies(seed))
