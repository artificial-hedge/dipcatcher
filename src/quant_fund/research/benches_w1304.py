"""Wave-1304 bench adapters: commonsense-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    boolq_studies,
    copa_studies,
    hellaswag_studies,
    openbookqa_studies,
    piqa_studies,
    siqa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13040


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


def bench_boolq_studies_family(seed: int = _SEED + 0):
    """boolq_studies: synthetic correctness bench."""
    return _finite_blob(boolq_studies.bench_boolq_studies(seed))


def bench_copa_studies_family(seed: int = _SEED + 1):
    """copa_studies: synthetic correctness bench."""
    return _finite_blob(copa_studies.bench_copa_studies(seed))


def bench_hellaswag_studies_family(seed: int = _SEED + 2):
    """hellaswag_studies: synthetic correctness bench."""
    return _finite_blob(hellaswag_studies.bench_hellaswag_studies(seed))


def bench_openbookqa_studies_family(seed: int = _SEED + 3):
    """openbookqa_studies: synthetic correctness bench."""
    return _finite_blob(openbookqa_studies.bench_openbookqa_studies(seed))


def bench_piqa_studies_family(seed: int = _SEED + 4):
    """piqa_studies: synthetic correctness bench."""
    return _finite_blob(piqa_studies.bench_piqa_studies(seed))


def bench_siqa_studies_family(seed: int = _SEED + 5):
    """siqa_studies: synthetic correctness bench."""
    return _finite_blob(siqa_studies.bench_siqa_studies(seed))
