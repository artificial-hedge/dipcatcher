"""Wave-1604 bench adapters: burrow-mammal canon (SYNTHETIC only)."""

from quant_fund.models import (
    dassie_qa_studies,
    gopher_qa_studies,
    mole_qa_studies,
    rabbit_qa_studies,
    shrew_qa_studies,
    springhare_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16040


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


def bench_dassie_qa_studies_family(seed: int = _SEED + 0):
    """dassie_qa_studies: synthetic correctness bench."""
    return _finite_blob(dassie_qa_studies.bench_dassie_qa_studies(seed))


def bench_gopher_qa_studies_family(seed: int = _SEED + 1):
    """gopher_qa_studies: synthetic correctness bench."""
    return _finite_blob(gopher_qa_studies.bench_gopher_qa_studies(seed))


def bench_mole_qa_studies_family(seed: int = _SEED + 2):
    """mole_qa_studies: synthetic correctness bench."""
    return _finite_blob(mole_qa_studies.bench_mole_qa_studies(seed))


def bench_rabbit_qa_studies_family(seed: int = _SEED + 3):
    """rabbit_qa_studies: synthetic correctness bench."""
    return _finite_blob(rabbit_qa_studies.bench_rabbit_qa_studies(seed))


def bench_shrew_qa_studies_family(seed: int = _SEED + 4):
    """shrew_qa_studies: synthetic correctness bench."""
    return _finite_blob(shrew_qa_studies.bench_shrew_qa_studies(seed))


def bench_springhare_qa_studies_family(seed: int = _SEED + 5):
    """springhare_qa_studies: synthetic correctness bench."""
    return _finite_blob(springhare_qa_studies.bench_springhare_qa_studies(seed))
