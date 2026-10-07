"""Wave-1615 bench adapters: lemur-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    golden_brown_qa_studies,
    gray_mouse_qa_studies,
    pygmy_qa_studies,
    slender_qa_studies,
    slow_qa_studies,
    thin_spined_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16150


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


def bench_golden_brown_qa_studies_family(seed: int = _SEED + 0):
    """golden_brown_qa_studies: synthetic correctness bench."""
    return _finite_blob(golden_brown_qa_studies.bench_golden_brown_qa_studies(seed))


def bench_gray_mouse_qa_studies_family(seed: int = _SEED + 1):
    """gray_mouse_qa_studies: synthetic correctness bench."""
    return _finite_blob(gray_mouse_qa_studies.bench_gray_mouse_qa_studies(seed))


def bench_pygmy_qa_studies_family(seed: int = _SEED + 2):
    """pygmy_qa_studies: synthetic correctness bench."""
    return _finite_blob(pygmy_qa_studies.bench_pygmy_qa_studies(seed))


def bench_slender_qa_studies_family(seed: int = _SEED + 3):
    """slender_qa_studies: synthetic correctness bench."""
    return _finite_blob(slender_qa_studies.bench_slender_qa_studies(seed))


def bench_slow_qa_studies_family(seed: int = _SEED + 4):
    """slow_qa_studies: synthetic correctness bench."""
    return _finite_blob(slow_qa_studies.bench_slow_qa_studies(seed))


def bench_thin_spined_qa_studies_family(seed: int = _SEED + 5):
    """thin_spined_qa_studies: synthetic correctness bench."""
    return _finite_blob(thin_spined_qa_studies.bench_thin_spined_qa_studies(seed))
