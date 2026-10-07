"""Wave-1531 bench adapters: woodpecker canon (SYNTHETIC only)."""

from quant_fund.models import (
    downy_qa_studies,
    flicker_qa_studies,
    pileated_qa_studies,
    sapsucker_qa_studies,
    woodpecker_qa_studies,
    wryneck_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15310


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


def bench_downy_qa_studies_family(seed: int = _SEED + 0):
    """downy_qa_studies: synthetic correctness bench."""
    return _finite_blob(downy_qa_studies.bench_downy_qa_studies(seed))


def bench_flicker_qa_studies_family(seed: int = _SEED + 1):
    """flicker_qa_studies: synthetic correctness bench."""
    return _finite_blob(flicker_qa_studies.bench_flicker_qa_studies(seed))


def bench_pileated_qa_studies_family(seed: int = _SEED + 2):
    """pileated_qa_studies: synthetic correctness bench."""
    return _finite_blob(pileated_qa_studies.bench_pileated_qa_studies(seed))


def bench_sapsucker_qa_studies_family(seed: int = _SEED + 3):
    """sapsucker_qa_studies: synthetic correctness bench."""
    return _finite_blob(sapsucker_qa_studies.bench_sapsucker_qa_studies(seed))


def bench_woodpecker_qa_studies_family(seed: int = _SEED + 4):
    """woodpecker_qa_studies: synthetic correctness bench."""
    return _finite_blob(woodpecker_qa_studies.bench_woodpecker_qa_studies(seed))


def bench_wryneck_qa_studies_family(seed: int = _SEED + 5):
    """wryneck_qa_studies: synthetic correctness bench."""
    return _finite_blob(wryneck_qa_studies.bench_wryneck_qa_studies(seed))
