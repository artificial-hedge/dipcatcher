"""Wave-1440 bench adapters: flora canon (SYNTHETIC only)."""

from quant_fund.models import (
    bamboo_qa_studies,
    cactus_qa_studies,
    fern_qa_studies,
    moss_qa_studies,
    pine_qa_studies,
    vine_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14400


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


def bench_bamboo_qa_studies_family(seed: int = _SEED + 0):
    """bamboo_qa_studies: synthetic correctness bench."""
    return _finite_blob(bamboo_qa_studies.bench_bamboo_qa_studies(seed))


def bench_cactus_qa_studies_family(seed: int = _SEED + 1):
    """cactus_qa_studies: synthetic correctness bench."""
    return _finite_blob(cactus_qa_studies.bench_cactus_qa_studies(seed))


def bench_fern_qa_studies_family(seed: int = _SEED + 2):
    """fern_qa_studies: synthetic correctness bench."""
    return _finite_blob(fern_qa_studies.bench_fern_qa_studies(seed))


def bench_moss_qa_studies_family(seed: int = _SEED + 3):
    """moss_qa_studies: synthetic correctness bench."""
    return _finite_blob(moss_qa_studies.bench_moss_qa_studies(seed))


def bench_pine_qa_studies_family(seed: int = _SEED + 4):
    """pine_qa_studies: synthetic correctness bench."""
    return _finite_blob(pine_qa_studies.bench_pine_qa_studies(seed))


def bench_vine_qa_studies_family(seed: int = _SEED + 5):
    """vine_qa_studies: synthetic correctness bench."""
    return _finite_blob(vine_qa_studies.bench_vine_qa_studies(seed))
