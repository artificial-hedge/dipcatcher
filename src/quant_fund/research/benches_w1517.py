"""Wave-1517 bench adapters: dragonfly canon (SYNTHETIC only)."""

from quant_fund.models import (
    clubtail_qa_studies,
    damselfly_qa_studies,
    darner_qa_studies,
    forktail_qa_studies,
    hawker_qa_studies,
    spreadwing_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15170


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


def bench_clubtail_qa_studies_family(seed: int = _SEED + 0):
    """clubtail_qa_studies: synthetic correctness bench."""
    return _finite_blob(clubtail_qa_studies.bench_clubtail_qa_studies(seed))


def bench_damselfly_qa_studies_family(seed: int = _SEED + 1):
    """damselfly_qa_studies: synthetic correctness bench."""
    return _finite_blob(damselfly_qa_studies.bench_damselfly_qa_studies(seed))


def bench_darner_qa_studies_family(seed: int = _SEED + 2):
    """darner_qa_studies: synthetic correctness bench."""
    return _finite_blob(darner_qa_studies.bench_darner_qa_studies(seed))


def bench_forktail_qa_studies_family(seed: int = _SEED + 3):
    """forktail_qa_studies: synthetic correctness bench."""
    return _finite_blob(forktail_qa_studies.bench_forktail_qa_studies(seed))


def bench_hawker_qa_studies_family(seed: int = _SEED + 4):
    """hawker_qa_studies: synthetic correctness bench."""
    return _finite_blob(hawker_qa_studies.bench_hawker_qa_studies(seed))


def bench_spreadwing_qa_studies_family(seed: int = _SEED + 5):
    """spreadwing_qa_studies: synthetic correctness bench."""
    return _finite_blob(spreadwing_qa_studies.bench_spreadwing_qa_studies(seed))
