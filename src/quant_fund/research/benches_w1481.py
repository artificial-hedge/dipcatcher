"""Wave-1481 bench adapters: arthropod canon (SYNTHETIC only)."""

from quant_fund.models import (
    earwig_qa_studies,
    katydid_qa_studies,
    mayfly_qa_studies,
    stonefly_qa_studies,
    wasp_qa_studies,
    weevil_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14810


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


def bench_earwig_qa_studies_family(seed: int = _SEED + 0):
    """earwig_qa_studies: synthetic correctness bench."""
    return _finite_blob(earwig_qa_studies.bench_earwig_qa_studies(seed))


def bench_katydid_qa_studies_family(seed: int = _SEED + 1):
    """katydid_qa_studies: synthetic correctness bench."""
    return _finite_blob(katydid_qa_studies.bench_katydid_qa_studies(seed))


def bench_mayfly_qa_studies_family(seed: int = _SEED + 2):
    """mayfly_qa_studies: synthetic correctness bench."""
    return _finite_blob(mayfly_qa_studies.bench_mayfly_qa_studies(seed))


def bench_stonefly_qa_studies_family(seed: int = _SEED + 3):
    """stonefly_qa_studies: synthetic correctness bench."""
    return _finite_blob(stonefly_qa_studies.bench_stonefly_qa_studies(seed))


def bench_wasp_qa_studies_family(seed: int = _SEED + 4):
    """wasp_qa_studies: synthetic correctness bench."""
    return _finite_blob(wasp_qa_studies.bench_wasp_qa_studies(seed))


def bench_weevil_qa_studies_family(seed: int = _SEED + 5):
    """weevil_qa_studies: synthetic correctness bench."""
    return _finite_blob(weevil_qa_studies.bench_weevil_qa_studies(seed))
