"""Wave-1541 bench adapters: pelagic canon (SYNTHETIC only)."""

from quant_fund.models import (
    anhinga_qa_studies,
    darter_qa_studies,
    diving_petrel_qa_studies,
    gadfly_qa_studies,
    manx_qa_studies,
    mollymawk_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15410


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


def bench_anhinga_qa_studies_family(seed: int = _SEED + 0):
    """anhinga_qa_studies: synthetic correctness bench."""
    return _finite_blob(anhinga_qa_studies.bench_anhinga_qa_studies(seed))


def bench_darter_qa_studies_family(seed: int = _SEED + 1):
    """darter_qa_studies: synthetic correctness bench."""
    return _finite_blob(darter_qa_studies.bench_darter_qa_studies(seed))


def bench_diving_petrel_qa_studies_family(seed: int = _SEED + 2):
    """diving_petrel_qa_studies: synthetic correctness bench."""
    return _finite_blob(diving_petrel_qa_studies.bench_diving_petrel_qa_studies(seed))


def bench_gadfly_qa_studies_family(seed: int = _SEED + 3):
    """gadfly_qa_studies: synthetic correctness bench."""
    return _finite_blob(gadfly_qa_studies.bench_gadfly_qa_studies(seed))


def bench_manx_qa_studies_family(seed: int = _SEED + 4):
    """manx_qa_studies: synthetic correctness bench."""
    return _finite_blob(manx_qa_studies.bench_manx_qa_studies(seed))


def bench_mollymawk_qa_studies_family(seed: int = _SEED + 5):
    """mollymawk_qa_studies: synthetic correctness bench."""
    return _finite_blob(mollymawk_qa_studies.bench_mollymawk_qa_studies(seed))
