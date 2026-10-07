"""Wave-1522 bench adapters: succulent canon (SYNTHETIC only)."""

from quant_fund.models import (
    agave_qa_studies,
    aloe_qa_studies,
    echeveria_qa_studies,
    haworthia_qa_studies,
    lithops_qa_studies,
    sedum_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15220


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


def bench_agave_qa_studies_family(seed: int = _SEED + 0):
    """agave_qa_studies: synthetic correctness bench."""
    return _finite_blob(agave_qa_studies.bench_agave_qa_studies(seed))


def bench_aloe_qa_studies_family(seed: int = _SEED + 1):
    """aloe_qa_studies: synthetic correctness bench."""
    return _finite_blob(aloe_qa_studies.bench_aloe_qa_studies(seed))


def bench_echeveria_qa_studies_family(seed: int = _SEED + 2):
    """echeveria_qa_studies: synthetic correctness bench."""
    return _finite_blob(echeveria_qa_studies.bench_echeveria_qa_studies(seed))


def bench_haworthia_qa_studies_family(seed: int = _SEED + 3):
    """haworthia_qa_studies: synthetic correctness bench."""
    return _finite_blob(haworthia_qa_studies.bench_haworthia_qa_studies(seed))


def bench_lithops_qa_studies_family(seed: int = _SEED + 4):
    """lithops_qa_studies: synthetic correctness bench."""
    return _finite_blob(lithops_qa_studies.bench_lithops_qa_studies(seed))


def bench_sedum_qa_studies_family(seed: int = _SEED + 5):
    """sedum_qa_studies: synthetic correctness bench."""
    return _finite_blob(sedum_qa_studies.bench_sedum_qa_studies(seed))
