"""Wave-1419 bench adapters: narrative-genre canon (SYNTHETIC only)."""

from quant_fund.models import (
    anecdote_qa_studies,
    ballad_qa_studies,
    biography_qa_studies,
    chronicle_qa_studies,
    epic_qa_studies,
    fable_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14190


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


def bench_anecdote_qa_studies_family(seed: int = _SEED + 0):
    """anecdote_qa_studies: synthetic correctness bench."""
    return _finite_blob(anecdote_qa_studies.bench_anecdote_qa_studies(seed))


def bench_ballad_qa_studies_family(seed: int = _SEED + 1):
    """ballad_qa_studies: synthetic correctness bench."""
    return _finite_blob(ballad_qa_studies.bench_ballad_qa_studies(seed))


def bench_biography_qa_studies_family(seed: int = _SEED + 2):
    """biography_qa_studies: synthetic correctness bench."""
    return _finite_blob(biography_qa_studies.bench_biography_qa_studies(seed))


def bench_chronicle_qa_studies_family(seed: int = _SEED + 3):
    """chronicle_qa_studies: synthetic correctness bench."""
    return _finite_blob(chronicle_qa_studies.bench_chronicle_qa_studies(seed))


def bench_epic_qa_studies_family(seed: int = _SEED + 4):
    """epic_qa_studies: synthetic correctness bench."""
    return _finite_blob(epic_qa_studies.bench_epic_qa_studies(seed))


def bench_fable_qa_studies_family(seed: int = _SEED + 5):
    """fable_qa_studies: synthetic correctness bench."""
    return _finite_blob(fable_qa_studies.bench_fable_qa_studies(seed))
