"""Wave-1507 bench adapters: raptor-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    buzzard_qa_studies,
    caracara_qa_studies,
    goshawk_qa_studies,
    merlin_qa_studies,
    peregrine_qa_studies,
    sparrowhawk_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15070


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


def bench_buzzard_qa_studies_family(seed: int = _SEED + 0):
    """buzzard_qa_studies: synthetic correctness bench."""
    return _finite_blob(buzzard_qa_studies.bench_buzzard_qa_studies(seed))


def bench_caracara_qa_studies_family(seed: int = _SEED + 1):
    """caracara_qa_studies: synthetic correctness bench."""
    return _finite_blob(caracara_qa_studies.bench_caracara_qa_studies(seed))


def bench_goshawk_qa_studies_family(seed: int = _SEED + 2):
    """goshawk_qa_studies: synthetic correctness bench."""
    return _finite_blob(goshawk_qa_studies.bench_goshawk_qa_studies(seed))


def bench_merlin_qa_studies_family(seed: int = _SEED + 3):
    """merlin_qa_studies: synthetic correctness bench."""
    return _finite_blob(merlin_qa_studies.bench_merlin_qa_studies(seed))


def bench_peregrine_qa_studies_family(seed: int = _SEED + 4):
    """peregrine_qa_studies: synthetic correctness bench."""
    return _finite_blob(peregrine_qa_studies.bench_peregrine_qa_studies(seed))


def bench_sparrowhawk_qa_studies_family(seed: int = _SEED + 5):
    """sparrowhawk_qa_studies: synthetic correctness bench."""
    return _finite_blob(sparrowhawk_qa_studies.bench_sparrowhawk_qa_studies(seed))
